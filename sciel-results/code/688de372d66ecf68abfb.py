"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    should equal 1 for all (i,j,k)."""
    rank = len(U[0]) if U else 0
    # Convert to numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                # Compute sum over c of u[c] * v[c] * w[c]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(n, m, p, rank, time_limit=10.0, seed=42):
    """Try to find a rank-decomposition of the given rank using randomized initialization
    and iterative optimization (tensor train / ALS-like approach).
    Returns (U, V, W) as lists if successful, None otherwise."""
    if rank < max(n, m, p):
        return None  # Lower bound
    
    start = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize U, V, W randomly
    U = rng.randn(n * m, rank).astype(float)
    V = rng.randn(m * p, rank).astype(float)
    W = rng.randn(p * n, rank).astype(float)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    max_iters = 500
    tol = 1e-10
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Alternating least squares style update
        # Fix V, W, solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = sol
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Fix U, W, solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = sol
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        # Fix U, V, solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = sol
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        U, V, W = U_new, V_new, W_new
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    # Check if we found an exact solution
    U_final = best_U
    V_final = best_V
    W_final = best_W
    
    # Verify with exact check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_final[i * m + j]
                v_val = V_final[j * p + k]
                w_val = W_final[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-8:
                    return None
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return U_list, V_list, W_list

def construct():
    n, m, p = DIMENSIONS
    
    # Heuristic lower bound: start from a more realistic rank
    # For 2x5x6, rank is likely around 25-35
    lower_bound = 20
    
    # Start from lower_bound and go up
    deadline = time.time() + 130  # Safety margin for the 150s limit
    
    # Search ranks from 20 to 40
    for rank in range(lower_bound, 41):
        if time.time() > deadline:
            break
        
        # Try a few restarts for each rank
        for attempt in range(5):
            if time.time() > deadline:
                break
            result = _try_rank(n, m, p, rank, time_limit=5.0, seed=rank * 100 + attempt)
            if result is not None:
                U, V, W = result
                # Final verification
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
