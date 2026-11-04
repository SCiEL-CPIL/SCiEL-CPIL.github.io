import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, tol=1e-9):
    """Verify exact decomposition."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _is_half_int(x, tol=1e-9):
    """Check if x is integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < tol

def _snap_half(arr):
    """Snap to nearest half-integer."""
    return np.round(np.array(arr, dtype=float) * 2) / 2.0

def _schoolbook():
    """Schoolbook decomposition, rank = n*m*p = 90."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    
    return U, V, W

def _compute_error(U, V, W):
    """Compute the sum of squared errors."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                err += (val - 1.0) ** 2
    return err

def _optimize_rank(r, time_budget, start_time):
    """Try to find a rank-r decomposition via optimization."""
    n, m, p = DIMENSIONS
    deadline = start_time + time_budget
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    np.random.seed(42)
    
    # We will try a few random initializations
    for trial in range(5):
        if time.time() > deadline - 2:
            break
            
        # Random initialization
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        # Flatten for scipy
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat = x[:n*m*r]
            V_flat = x[n*m*r : n*m*r + m*p*r]
            W_flat = x[n*m*r + m*p*r :]
            
            U_mat = U_flat.reshape(n*m, r)
            V_mat = V_flat.reshape(m*p, r)
            W_mat = W_flat.reshape(p*n, r)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_mat[i*m+j] * V_mat[j*p+k] * W_mat[k*n+i])
                        err += (val - 1.0) ** 2
            return err

        # Use L-BFGS-B
        try:
            res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxls': 10})
            
            if res.success or res.fun < best_err:
                x_opt = res.x
                U_opt = x_opt[:n*m*r].reshape(n*m, r)
                V_opt = x_opt[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_opt = x_opt[n*m*r + m*p*r :].reshape(p*n, r)
                
                err = res.fun
                if err < best_err:
                    best_err = err
                    best_U = U_opt
                    best_V = V_opt
                    best_W = W_opt
        except Exception:
            continue

        if best_err < 1e-6:
            break

    if best_err < 1e-5 and best_U is not None:
        # Snap to half-integers
        U_snap = _snap_half(best_U)
        V_snap = _snap_half(best_V)
        W_snap = _snap_half(best_W)
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if _verify(U_list, V_list, W_list):
            return U_list, V_list, W_list
    
    return None

def _try_structured_rank(r, time_budget, start_time):
    """Try a structured approach for specific ranks."""
    n, m, p = DIMENSIONS
    deadline = start_time + time_budget
    
    # For rank = n*m*p, use schoolbook
    if r == n * m * p:
        return _schoolbook()
    
    # Try random restarts with optimization
    return _optimize_rank(r, time_budget, start_time)

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # We try a descending list of ranks to find the best (lowest) rank we can.
    # If we find a valid decomposition for rank R, we update best_rank and continue
    # to see if we can find an even lower rank.
    
    ranks_to_try = [89, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        if rank >= best_rank:
            break
        
        # Allocate time
        time_alloc = 8
        
        result = _try_structured_rank(rank, time_alloc, start_time)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
