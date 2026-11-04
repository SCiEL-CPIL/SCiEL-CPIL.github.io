"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Build the tensor T[i,j,k] = 1 for all valid indices (the multiplication tensor)
    # T is a 3D array of shape (n, m, p)
    T = np.ones((n, m, p), dtype=float)
    
    # Try to find a lower-rank decomposition
    # A rank-r decomposition: T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W is indexed as W[k*n+i, r] per the trace convention
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: the tensor has 72 entries, each rank-1 term contributes to all 72 entries
    # But we need to satisfy all 72 equations. Minimum rank is at least ceil(72 / max_terms_per_rank)
    # Each rank-1 term has 9*24*24 = 5184 parameters but with scaling, effectively 5184-2=5182
    # This is not helpful. Let's just try ranks from 20 up to 72.
    
    # Actually, let's use a smarter approach: use the fact that the tensor is separable in a certain way
    # For matrix multiplication, T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k} is not right.
    # The correct tensor is T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] involves all pairs)
    # Wait, no. The multiplication tensor for (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k, and the bilinear form is sum_{i,j,k} T[i,j,k] * A[i,j]*B[j,k]*C[k,i]
    # Actually, the standard convention: the tensor T in (R^n*)^i ⊗ (R^m*)^j ⊗ (R^p*)^k with T[i,j,k] = 1
    # and the decomposition is T = sum_r u_r ⊗ v_r ⊗ w_r where:
    #   u_r is in R^{n*m} (indexed by (i,j)), v_r in R^{m*p} (indexed by (j,k)), w_r in R^{p*n} (indexed by (k,i))
    # and T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try a random initialization approach for various ranks
    for R in range(20, rank_schoolbook + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W randomly
            U_rand = np.random.randn(n * m, R)
            V_rand = np.random.randn(m * p, R)
            W_rand = np.random.randn(p * n, R)
            
            # We need to solve: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a system of 72 equations in 57*R unknowns (but with scaling ambiguity)
            
            # Let's use a simple iterative approach: fix two factors, solve for the third
            # This is like alternating least squares for tensor decomposition
            
            converged = False
            for iteration in range(1000):
                if time.time() > deadline:
                    break
                    
                # Update U: for each (i,j), we need sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                # This is a system: for fixed (i,j), let M_k = V[j*p+k, r] * W[k*n+i, r] for r=1..R
                # We need: sum_r U[i*m+j, r] * M_k(r) = 1 for all k
                # This is a linear system in U[i*m+j, :] of size R x p (but p=8, R>=20, so underdetermined)
                # We want the minimum norm solution: U[i*m+j, :] = M^T (M M^T)^{-1} * 1
                
                U_new = np.zeros_like(U_rand)
                for i in range(n):
                    for j in range(m):
                        # M is R x p matrix where M[r, k] = V_rand[j*p+k, r] * W_rand[k*n+i, r]
                        M = np.zeros((R, p))
                        for k in range(p):
                            M[:, k] = V_rand[j*p+k, :] * W_rand[k*n+i, :]
                        # Solve M^T * U_row = 1 (vector of ones of length p)
                        # Minimum norm: U_row = M @ (M.T @ M)^{-1} @ 1
                        # But M is R x p with R > p, so M.T @ M is p x p
                        try:
                            MtM = M.T @ M
                            # Add regularization for numerical stability
                            MtM += 1e-10 * np.eye(p)
                            U_row = np.linalg.solve(MtM, np.ones(p))
                            U_new[i*m+j, :] = M @ U_row
                        except np.linalg.LinAlgError:
                            U_new[i*m+j, :] = U_rand[i*m+j, :]
                
                # Update V: for each (j,k), we need sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for all i
                V_new = np.zeros_like(V_rand)
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((R, n))
                        for i in range(n):
                            M[:, i] = U_new[i*m+j, :] * W_rand[k*n+i, :]
                        try:
                            MtM = M.T @ M
                            MtM += 1e-10 * np.eye(n)
                            V_row = np.linalg.solve(MtM, np.ones(n))
                            V_new[j*p+k, :] = M @ V_row
                        except np.linalg.LinAlgError:
                            V_new[j*p+k, :] = V_rand[j*p+k, :]
                
                # Update W: for each (k,i), we need sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all j
                W_new = np.zeros_like(W_rand)
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((R, m))
                        for j in range(m):
                            M[:, j] = U_new[i*m+j, :] * V_new[j*p+k, :]
                        try:
                            MtM = M.T @ M
                            MtM += 1e-10 * np.eye(m)
                            W_row = np.linalg.solve(MtM, np.ones(m))
                            W_new[k*n+i, :] = M @ W_row
                        except np.linalg.LinAlgError:
                            W_new[k*n+i, :] = W_rand[k*n+i, :]
                
                # Check convergence
                diff = np.max(np.abs(U_new - U_rand)) + np.max(np.abs(V_new - V_rand)) + np.max(np.abs(W_new - W_rand))
                U_rand, V_rand, W_rand = U_new, V_new, W_new
                if diff < 1e-10:
                    converged = True
                    break
            
            if time.time() > deadline:
                break
                
            # Check if we have an exact decomposition
            # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_rand[i*m+j, :] * V_rand[j*p+k, :] * W_rand[k*n+i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                # We found a rank-R decomposition!
                best_rank = R
                # Convert to list format
                best_U = U_rand.tolist()
                best_V = V_rand.tolist()
                best_W = W_rand.tolist()
                # Round to reduce numerical noise (but keep enough precision)
                # Actually, we should check if rounding to simple fractions works
                # For now, just use the float values
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # If we found a better decomposition, return it; otherwise return schoolbook
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
