"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # 15s margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Default: Schoolbook decomposition
    # Rank = n * m * p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find lower rank decompositions
    # The tensor T_{i,j,k} = delta_{i,k} (for matrix mult A[i,j]*B[j,k])
    # We want: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Target: find rank r < 105 such that there exist U,V,W with:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if i==k else 0
    
    # Strategy: Use random initialization and gradient-free optimization
    # to minimize the residual of the tensor decomposition
    
    def check_decomposition(U_mat, V_mat, W_mat, tol=1e-6):
        """Check if U, V, W form a valid decomposition."""
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    expected = 1.0 if i == k else 0.0
                    if abs(val - expected) > tol:
                        return False
        return True
    
    def random_search(rank_target, max_iterations=10000):
        """Try random initialization with local refinement."""
        U_mat = np.random.randn(n*m, rank_target) * 0.5
        V_mat = np.random.randn(m*p, rank_target) * 0.5
        W_mat = np.random.randn(p*n, rank_target) * 0.5
        
        # Simple coordinate descent to minimize residual
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            improved = False
            for r in range(rank_target):
                # Optimize U[:, r]
                old_U = U_mat[:, r].copy()
                best_val = np.inf
                for trial in range(3):
                    if trial == 0:
                        candidate = old_U
                    else:
                        candidate = old_U + np.random.randn(n*m) * 0.1
                    # Compute residual contribution
                    res = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                expected = 1.0 if i == k else 0.0
                                prod = candidate[i*m+j] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                                # Sum over other r'
                                total = prod
                                for r2 in range(rank_target):
                                    if r2 != r:
                                        total += U_mat[i*m+j, r2] * V_mat[j*p+k, r2] * W_mat[k*n+i, r2]
                                res += (total - expected) ** 2
                    if res < best_val:
                        best_val = res
                        best_candidate = candidate.copy()
                if best_val < np.inf:
                    U_mat[:, r] = best_candidate
                    improved = True
                
                # Optimize V[:, r]
                old_V = V_mat[:, r].copy()
                best_val = np.inf
                for trial in range(3):
                    if trial == 0:
                        candidate = old_V
                    else:
                        candidate = old_V + np.random.randn(m*p) * 0.1
                    res = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                expected = 1.0 if i == k else 0.0
                                prod = U_mat[i*m+j, r] * candidate[j*p+k] * W_mat[k*n+i, r]
                                total = prod
                                for r2 in range(rank_target):
                                    if r2 != r:
                                        total += U_mat[i*m+j, r2] * V_mat[j*p+k, r2] * W_mat[k*n+i, r2]
                                res += (total - expected) ** 2
                    if res < best_val:
                        best_val = res
                        best_candidate = candidate.copy()
                if best_val < np.inf:
                    V_mat[:, r] = best_candidate
                    improved = True
                
                # Optimize W[:, r]
                old_W = W_mat[:, r].copy()
                best_val = np.inf
                for trial in range(3):
                    if trial == 0:
                        candidate = old_W
                    else:
                        candidate = old_W + np.random.randn(p*n) * 0.1
                    res = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                expected = 1.0 if i == k else 0.0
                                prod = U_mat[i*m+j, r] * V_mat[j*p+k, r] * candidate[k*n+i]
                                total = prod
                                for r2 in range(rank_target):
                                    if r2 != r:
                                        total += U_mat[i*m+j, r2] * V_mat[j*p+k, r2] * W_mat[k*n+i, r2]
                                res += (total - expected) ** 2
                    if res < best_val:
                        best_val = res
                        best_candidate = candidate.copy()
                if best_val < np.inf:
                    W_mat[:, r] = best_candidate
                    improved = True
            
            if not improved:
                break
        
        # Check if we found a valid decomposition
        if check_decomposition(U_mat, V_mat, W_mat, tol=1e-4):
            # Convert to list format with rounded values
            U_list = [[round(float(x), 6) for x in U_mat[:, r]] for r in range(rank_target)]
            V_list = [[round(float(x), 6) for x in V_mat[:, r]] for r in range(rank_target)]
            W_list = [[round(float(x), 6) for x in W_mat[:, r]] for r in range(rank_target)]
            
            # Transpose to get correct shape: (n*m, rank)
            U_final = [[U_list[r][i] for r in range(rank_target)] for i in range(n*m)]
            V_final = [[V_list[r][i] for r in range(rank_target)] for i in range(m*p)]
            W_final = [[W_list[r][i] for r in range(rank_target)] for i in range(p*n)]
            
            return U_final, V_final, W_final
        
        return None
    
    # Try ranks from 90 down to 70
    for r_target in range(90, 69, -1):
        if time.time() > deadline:
            break
        result = random_search(r_target, max_iterations=5000)
        if result is not None:
            U_final, V_final, W_final = result
            best_U, best_V, best_W = U_final, V_final, W_final
            best_rank = r_target
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
