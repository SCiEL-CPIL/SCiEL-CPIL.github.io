"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    rank = U.shape[1]
    
    if U.shape != (n_rows, rank) or V.shape != (m_rows, rank) or W.shape != (p_rows, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                product_sum = np.sum(u_val * v_val * w_val)
                if abs(product_sum - 1.0) > tol:
                    return False
    return True

def _als_decompose(n, m, p, R, max_iter=200, tol=1e-12, seed=42):
    """
    Alternating Least Squares for tensor decomposition of matrix multiplication.
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all valid (i,j,k).
    """
    np.random.seed(seed)
    
    U = np.random.randn(n * m, R) * 0.1
    V = np.random.randn(m * p, R) * 0.1
    W = np.random.randn(p * n, R) * 0.1
    
    for iteration in range(max_iter):
        # Update U: for each (i,j), find U[i*m+j,:] to minimize error
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                u, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U[i * m + j, :] = u
        
        # Update V: for each (j,k), find V[j*p+k,:]
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                v, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V[j * p + k, :] = v
        
        # Update W: for each (k,i), find W[k*n+i,:]
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W[k * n + i, :] = w
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < tol:
            break
    
    return U, V, W, max_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 128
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank
    
    # Start with schoolbook
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to find lower ranks using ALS
    # Lower bound: max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # Try ranks from 33 up to 127
    ranks_to_try = list(range(33, 128))
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        if R >= best_rank:
            continue
        
        # Try multiple seeds for this rank
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            U_trial, V_trial, W_trial, err = _als_decompose(
                n, m, p, R, max_iter=500, seed=seed, tol=1e-12
            )
            
            if err < 1e-9:
                # Verify
                if _verify_decomposition(U_trial, V_trial, W_trial, n, m, p, tol=1e-9):
                    best_U, best_V, best_W = U_trial.copy(), V_trial.copy(), W_trial.copy()
                    best_rank = R
                    print(f"Found valid rank {best_rank} with seed {seed}")
                    break  # Move to next lower rank
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
