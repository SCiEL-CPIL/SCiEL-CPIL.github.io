"""Matrix multiplication 2x5x6 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-8):
    n, m, p = DIMENSIONS
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the squared error of the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                err += (s - 1.0) ** 2
    return err

def _als_step(U, V, W, n, m, p, rank):
    """Perform one ALS update step. Returns new U, V, W."""
    U_new = np.zeros_like(U)
    V_new = np.zeros_like(V)
    W_new = np.zeros_like(W)
    
    # Update U
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
    
    # Update V
    for j in range(m):
        for k in range(p):
            M = np.zeros((n, rank))
            for i in range(n):
                M[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
            try:
                sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                V_new[j * p + k, :] = sol
            except:
                V_new[j * p + k, :] = V[j * p + k, :]
    
    # Update W
    for k in range(p):
        for i in range(n):
            M = np.zeros((m, rank))
            for j in range(m):
                M[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
            try:
                sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                W_new[k * n + i, :] = sol
            except:
                W_new[k * n + i, :] = W[k * n + i, :]
    
    return U_new, V_new, W_new

def _try_rank(n, m, p, rank, time_limit=10.0, seed=42):
    """Try to find a rank-decomposition of the given rank using randomized initialization
    and iterative optimization (ALS-like approach).
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
    
    max_iters = 200
    tol = 1e-10
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        U, V, W = _als_step(U, V, W, n, m, p, rank)
        
        # Compute error
        err = _compute_error(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < tol:
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
    
    # Lower bound on rank
    lower_bound = max(n, m, p)
    
    # Start from lower_bound and go up
    deadline = time.time() + 130  # Safety margin for the 150s limit
    
    for rank in range(lower_bound, n * m * p + 1):
        if time.time() > deadline:
            break
        
        for attempt in range(5):  # A few random restarts
            if time.time() > deadline:
                break
            result = _try_rank(n, m, p, rank, time_limit=20.0, seed=rank * 100 + attempt)
            if result is not None:
                U, V, W = result
                # Final verification
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    return _schoolbook()
# EVOLVE-BLOCK-END
