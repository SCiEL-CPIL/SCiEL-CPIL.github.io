"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    
    The multiplication tensor T has indices (i, j, k) where:
    - i in [0, n) = [0, 3)
    - j in [0, m) = [0, 4)  
    - k in [0, p) = [0, 6)
    
    T[i,j,k] = 1 if the product entry (i,k) of A@B involves A[i,j]*B[j,k]
    
    We seek rank-r decomposition:
    T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Strategy: Model the tensor decomposition as a sparse matrix factorization problem.
    Flatten the tensor into a matrix and apply ALS with hard thresholding to find
    integer/half-integer solutions.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 24
    pn = p * n  # 18
    
    # Build the multiplication tensor as a 3D array T[i,j,k]
    # T[i,j,k] = 1 for all valid (i,j,k)
    T = np.ones((n, m, p), dtype=float)
    
    def compute_error(U, V, W, n, m, p):
        """Compute the reconstruction error of the tensor."""
        r = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err += (val - 1.0) ** 2
        return np.sqrt(err)
    
    def try_rank_als(r, max_iter=300, tol=1e-10, seed=42):
        """Try to find a rank-r decomposition using ALS with hard thresholding."""
        rng = np.random.RandomState(seed)
        
        # Initialize U, V, W with random values
        U = rng.randn(nm, r) * 0.1
        V = rng.randn(mp, r) * 0.1
        W = rng.randn(pn, r) * 0.1
        
        for it in range(max_iter):
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # For this (i,j), we have T[i,j,:] of shape (p,)
                    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    # Let A_k,c = V[j*p+k, c] * W[k*n+i, c] for k in [0,p), c in [0,r)
                    A = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    # Solve: A^T @ U[i*m+j,:] = T[i,j,:]
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A.T, T[i, j, :], rcond=None)
                        U_new[i * m + j, :] = sol
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # For fixed (j,k): T[:,j,k] = V[j*p+k,:] * (U[i*m+j,c] * W[k*n+i,c])_c
                    B = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            B[i, c] = U[i * m + j, c] * W[k * n + i, c]
                    try:
                        sol, residuals, rank_B, sv = np.linalg.lstsq(B.T, T[:, j, k], rcond=None)
                        V_new[j * p + k, :] = sol
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # For fixed (k,i): T[i,:,k] = W[k*n+i,:] * (U[i*m+j,c] * V[j*p+k,c])_c
                    C = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            C[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    try:
                        sol, residuals, rank_C, sv = np.linalg.lstsq(C.T, T[i, :, k], rcond=None)
                        W_new[k * n + i, :] = sol
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
            W = W_new
            
            # Check convergence
            error = compute_error(U, V, W, n, m, p)
            if error < tol:
                break
        
        # Project to integer/half-integer values
        U_int = np.round(2 * U) / 2.0
        V_int = np.round(2 * V) / 2.0
        W_int = np.round(2 * W) / 2.0
        
        # Verify the projected solution
        err_int = compute_error(U_int, V_int, W_int, n, m, p)
        if err_int < 1e-6:
            return U_int, V_int, W_int, True
        return None, None, None, False
    
    # Try ranks from a lower bound up to schoolbook
    # For 3x4x6, try smaller ranks first
    schoolbook = n * m * p  # 72
    
    # Try several ranks with different seeds
    for r in range(20, min(40, schoolbook + 1)):
        for seed in range(5):
            U_int, V_int, W_int, success = try_rank_als(r, max_iter=100, tol=1e-8, seed=seed)
            if success:
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Fallback to schoolbook
    rank = schoolbook
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
