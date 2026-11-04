"""Matrix multiplication 3,4,8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 4, 8

    # Schoolbook fallback
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    # Verify a decomposition
    def verify(U, V, W, n, m, p, tol=1e-8):
        r = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        T = np.zeros((n, m, p))
        for c in range(r):
            uc = U_np[:, c]
            vc = V_np[:, c]
            wc = W_np[:, c]
            u3 = uc.reshape(n, m)
            v3 = vc.reshape(m, p)
            w3 = wc.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u3[i, j] * v3[j, k] * w3[k, i]
        return np.allclose(T, 1.0, atol=tol)

    # First, get the schoolbook solution
    U, V, W, rank = schoolbook(n, m, p)
    
    if not verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}
    
    # Now try to improve using group-theoretic orbit analysis
    # The key insight: the schoolbook decomposition has 96 terms, one for each (i,j,k) triple.
    # We can look for linear dependencies among these terms by analyzing the structure.
    
    # Represent each schoolbook term as a vector in the tensor space
    # Term (i,j,k) corresponds to:
    #   U_term[i*m+j] = 1, else 0
    #   V_term[j*p+k] = 1, else 0
    #   W_term[k*n+i] = 1, else 0
    
    # The tensor contribution of term c is:
    #   T_c[i,j,k] = U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    
    # For schoolbook, T_c[i,j,k] = 1 if (i,j,k) matches the term's indices, else 0.
    # So each schoolbook term contributes to exactly one entry of the tensor.
    
    # To reduce rank, we need to find if some terms can be combined.
    # This is equivalent to asking: can we find vectors u_c, v_c, w_c for c < r' < r
    # such that sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all (i,j,k)?
    
    # Strategy: Use the fact that the problem has symmetry.
    # The dimensions are n=3, m=4, p=8.
    # We can try to exploit the factorization p = 2*4.
    
    # Let k = 4*k1 + k2 where k1 in {0,1}, k2 in {0,1,2,3}.
    # Then the tensor splits into two blocks:
    #   T[i,j,4*k1+k2] = 1 for all i,j,k1,k2
    
    # If we can compute each 3x4x4 block with rank r_block,
    # the total rank is 2*r_block.
    
    # Schoolbook for 3x4x4: rank 48. Total: 96.
    
    # Can we do better for 3x4x4?
    # Known: rank of 2x2x2 is 7 (Strassen).
    # For 3x4x4, the rank is likely between 24 and 48.
    
    # Let's try a heuristic search for a lower-rank decomposition of 3x4x4.
    
    def find_3x4x4_decomposition(max_rank, time_budget):
        """Try to find a rank-r decomposition of 3x4x4 matrix multiplication."""
        n2, m2, p2 = 3, 4, 4
        
        # Start with schoolbook
        r = n2 * m2 * p2
        U2 = [[0] * r for _ in range(n2 * m2)]
        V2 = [[0] * r for _ in range(m2 * p2)]
        W2 = [[0] * r for _ in range(p2 * n2)]
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U2[i * m2 + j][col] = 1
                    V2[j * p2 + k][col] = 1
                    W2[k * n2 + i][col] = 1
                    col += 1
        
        # Try to reduce rank by removing terms and solving
        # This is a hard problem, so we use a simple heuristic
        
        # For now, just return schoolbook
        return U2, V2, W2, r
    
    # Given time constraints, let's try a more direct approach:
    # Use randomized search to find a lower-rank decomposition
    
    np.random.seed(42)
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try ranks from 95 down to 80
    for target_rank in range(95, 79, -1):
        if time.time() > deadline - 20:
            break
        
        # Try to find a rank-target_rank decomposition
        # Use a simple iterative approach
        
        # Initialize with random vectors
        U_rand = np.random.randn(n * m, target_rank) * 0.1
        V_rand = np.random.randn(m * p, target_rank) * 0.1
        W_rand = np.random.randn(p * n, target_rank) * 0.1
        
        # Iteratively improve
        for iteration in range(100):
            if time.time() > deadline - 15:
                break
            
            # Update W: for each (i,j,k), we need
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Fix U and V, solve for W
            
            # This is a linear system in W
            # For each (i,j,k): sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
            
            # Build the system
            A_system = np.zeros((n * m * p, p * n * target_rank))
            b_system = np.ones(n * m * p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            coeff = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                            A_system[idx, (k * n + i) * target_rank + c] = coeff
                        idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_system, b_system, rcond=None)
                W_rand = W_flat.reshape(p * n, target_rank)
            except:
                break
            
            # Check error
            T_recon = np.zeros((n, m, p))
            for c in range(target_rank):
                u3 = U_rand[:, c].reshape(n, m)
                v3 = V_rand[:, c].reshape(m, p)
                w3 = W_rand[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon[i, j, k] += u3[i, j] * v3[j, k] * w3[k, i]
            
            error = np.max(np.abs(T_recon - 1.0))
            
            if error < 1e-8:
                # Found a valid decomposition
                best_rank = target_rank
                best_U = U_rand.tolist()
                best_V = V_rand.tolist()
                best_W = W_rand.tolist()
                break
            
            # Update U and V similarly (simplified)
            # For now, just perturb
            U_rand += np.random.randn(n * m, target_rank) * 0.01
            V_rand += np.random.randn(m * p, target_rank) * 0.01
        
        if best_rank < rank:
            break
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
