"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _snap_to_half_int(arr):
    return [[round(x * 2) / 2 for x in row] for row in arr]

def _check_exact(U, V, W, T, tol=1e-9):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    R = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    err = np.sum((R - T)**2)
    return err < tol

def _try_rank_optimize(n, m, p, rank, T, deadline, restarts=20):
    nm, mp, pn = n * m, m * p, p * n
    U_shape, V_shape, W_shape = (nm, rank), (mp, rank), (pn, rank)
    size_u, size_v, size_w = nm * rank, mp * rank, pn * rank
    total_size = size_u + size_v + size_w
    
    def objective(x):
        U = x[:size_u].reshape(U_shape)
        V = x[size_u:size_u+size_v].reshape(V_shape)
        W = x[size_u+size_v:].reshape(W_shape)
        R = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((R - T)**2)
    
    def callback(xk):
        if time.time() > deadline:
            raise ValueError("Timeout")
    
    for r in range(restarts):
        if time.time() > deadline:
            break
        
        # Use structured initialization: start near schoolbook pattern
        if rank <= 24:
            # Initialize with random values but with some structure
            x0 = np.random.randn(total_size) * 0.5
        else:
            x0 = np.random.randn(total_size)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}, 
                          callback=callback)
        except ValueError:
            break
        except Exception:
            continue
        
        if res.success or (hasattr(res, 'fun') and res.fun < 1e-6):
            x_opt = res.x
            U = x_opt[:size_u].reshape(U_shape).tolist()
            V = x_opt[size_u:size_u+size_v].reshape(V_shape).tolist()
            W = x_opt[size_u+size_v:].reshape(W_shape).tolist()
            
            U_s, V_s, W_s = _snap_to_half_int(U), _snap_to_half_int(V), _snap_to_half_int(W)
            
            if _check_exact(U_s, V_s, W_s, T):
                return U_s, V_s, W_s
    return None

def _try_greedy_merge(n, m, p, T, deadline):
    """Try to merge schoolbook terms to reduce rank."""
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p  # 24
    
    # Build schoolbook
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to find linear dependencies among the columns
    # If sum_r c_r * col_r == 0 for U, V, W simultaneously, we can eliminate
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Stack all columns into one matrix to find null space
    # Each column r has entries in U[:,r], V[:,r], W[:,r]
    # We want to find c such that sum_r c_r * [U[:,r]; V[:,r]; W[:,r]] = 0
    stacked = np.vstack([U_np, V_np, W_np])  # (nm+mp+pn, rank)
    
    # Find null space of stacked matrix
    try:
        from scipy.linalg import null_space
        null = null_space(stacked, rcond=1e-10)
    except Exception:
        null = np.zeros((rank, 0))
    
    if null.shape[1] > 0:
        # We have dependencies - try to use them
        # For each null vector, we can eliminate one column
        # This is complex; for now, just return schoolbook
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0  # 20s safety margin
    
    T = _build_tensor(n, m, p)
    
    # Try ranks from 20 down to 16
    # Known: rank 24 is schoolbook, rank 20 is conjectured possible
    for rank in [20, 19, 18, 17, 16]:
        if time.time() > deadline:
            break
        
        result = _try_rank_optimize(n, m, p, rank, T, deadline, restarts=25)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Try greedy merge as backup
        if time.time() < deadline - 30:
            result = _try_greedy_merge(n, m, p, T, deadline)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": W}
    
    # Fallback to schoolbook rank 24
    rank = 24
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
