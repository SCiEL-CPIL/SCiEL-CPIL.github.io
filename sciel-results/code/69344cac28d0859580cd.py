import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _to_halfint_list(arr):
    """Convert numpy array to list with integer/half-integer entries."""
    result = []
    for row in arr:
        row_list = []
        for val in row:
            rounded = round(val * 2) / 2.0
            if abs(rounded - int(rounded)) < 1e-9:
                row_list.append(int(rounded))
            else:
                row_list.append(rounded)
        result.append(row_list)
    return result

def _compute_residuals(params, n, m, p, r):
    """Compute the residual vector for least_squares."""
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U + size_V]
    W_flat = params[size_U + size_V:]
    
    U = U_flat.reshape((n * m, r))
    V = V_flat.reshape((m * p, r))
    W = W_flat.reshape((p * n, r))
    
    U_3d = U.reshape((n, m, r))
    V_3d = V.reshape((m, p, r))
    W_3d = W.reshape((p, n, r))
    
    residuals = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            u_row = U_3d[i, j, :]
            for k in range(p):
                v_row = V_3d[j, k, :]
                w_row = W_3d[k, i, :]
                val = np.dot(u_row * v_row, w_row)
                residuals[i, j, k] = val - 1.0
                
    return residuals.flatten()

def _try_rank_scipy(r, n, m, p, max_seeds=20, time_budget=15.0):
    """Try to find a rank-r decomposition using scipy.optimize.least_squares."""
    start_time = time.time()
    
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    total_size = size_U + size_V + size_W
    
    # Use trf for underdetermined systems
    # trf can handle bounds, but we don't need them.
    # It is more robust than lm for N < M.
    
    for seed in range(max_seeds):
        if time.time() - start_time > time_budget:
            break
            
        np.random.seed(seed)
        
        # Different initialization strategies
        if seed % 3 == 0:
            params0 = np.random.uniform(-1, 1, total_size)
        elif seed % 3 == 1:
            params0 = np.random.normal(0, 0.5, total_size)
        else:
            # Sparse initialization
            params0 = np.random.uniform(-1, 1, total_size)
            mask = np.random.random(total_size) < 0.5
            params0[mask] = 0.0
            # Ensure not all zero
            if np.all(params0 == 0):
                params0[0] = 1.0

        try:
            result = opt.least_squares(
                _compute_residuals, 
                params0, 
                args=(n, m, p, r),
                method='trf',
                max_nfev=2000
            )
            
            if result.cost < 1e-10:
                U_flat = result.x[:size_U]
                V_flat = result.x[size_U:size_U + size_V]
                W_flat = result.x[size_U + size_V:]
                
                U = U_flat.reshape((n * m, r))
                V = V_flat.reshape((m * p, r))
                W = W_flat.reshape((p * n, r))
                
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    return U_rounded, V_rounded, W_rounded
                    
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140 # Safety margin for 150s limit
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    r_min = max(n*m, m*p, n*p)
    
    # Try ranks from r_min up to r_school - 1
    for r in range(r_min, rank_school):
        if time.time() > deadline - 10:
            break
            
        # Allocate time based on rank
        # Lower ranks are harder, give them more time? Or just uniform?
        # With 140s total, and ~70 ranks, ~2s per rank is tight.
        # Let's try 5s per rank for the first few, then less.
        time_per_rank = 5.0 if r < 50 else 2.0
        
        result = _try_rank_scipy(r, n, m, p, max_seeds=10, time_budget=time_per_rank)
        
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {
                "U": _to_halfint_list(U),
                "V": _to_halfint_list(V),
                "W": _to_halfint_list(W)
            }
            return best_result
            
    return best_result
