import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _is_integer_or_half_integer(arr, tol=1e-5):
    # Check if all elements are close to x.0 or x.5
    doubled = arr * 2
    rounded = np.round(doubled)
    if np.all(np.abs(doubled - rounded) < tol):
        return True
    return False

def _verify(U, V, W, n, m, p, tol=1e-6):
    if U.shape != (n*m, U.shape[1]) or V.shape != (m*p, V.shape[1]) or W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
        
    U_flat = U.reshape(-1, U.shape[1])
    V_flat = V.reshape(-1, V.shape[1])
    W_flat = W.reshape(-1, W.shape[1])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_flat[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_find_decomposition(n, m, p, target_rank, max_iters=500, time_limit=20.0):
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    num_vars = (nm + mp + pn) * target_rank
    num_res = n * m * p
    
    # Least squares 'lm' requires num_res >= num_vars
    # 'trf' can handle some underdetermined cases but is slower/less reliable for this
    if num_res < num_vars:
        return None
        
    best_error = float('inf')
    best_solution = None
    
    for restart in range(3):
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.randn(nm, target_rank) * 0.5
        V = np.random.randn(mp, target_rank) * 0.5
        W = np.random.randn(pn, target_rank) * 0.5
        
        def objective(x):
            U_flat_len = nm * target_rank
            V_flat_len = mp * target_rank
            
            U_r = x[0:U_flat_len].reshape(nm, target_rank)
            V_r = x[U_flat_len:U_flat_len+V_flat_len].reshape(mp, target_rank)
            W_r = x[U_flat_len+V_flat_len:].reshape(pn, target_rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                        res.append(val - 1.0)
            return res
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            # Use 'trf' as it is more robust, but check bounds? No bounds needed for exact decompositions usually.
            # However, 'lm' is faster if overdetermined. We checked num_res >= num_vars.
            result = least_squares(objective, x0, method='lm', max_nfev=max_iters)
            if result.cost < best_error:
                best_error = result.cost
                if best_error < 1e-8:
                    best_solution = result.x
                    break
        except Exception:
            continue
            
    if best_solution is not None:
        U_flat_len = nm * target_rank
        V_flat_len = mp * target_rank
        U_out = best_solution[0:U_flat_len].reshape(nm, target_rank)
        V_out = best_solution[U_flat_len:U_flat_len+V_flat_len].reshape(mp, target_rank)
        W_out = best_solution[U_flat_len+V_flat_len:].reshape(pn, target_rank)
        
        # Check integrality/half-integrality
        if _is_integer_or_half_integer(U_out) and _is_integer_or_half_integer(V_out) and _is_integer_or_half_integer(W_out):
            if _verify(U_out, V_out, W_out, n, m, p):
                return {"U": U_out, "V": V_out, "W": W_out}
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_rank_344 = 48
    best_decomp_344 = _schoolbook(3, 4, 4)
    
    # Try to improve rank for 3x4x4
    # Schoolbook rank is 48.
    # We need num_res (48) >= num_vars (40 * R).
    # So 40 * R <= 48 => R <= 1.
    # So we can only try R=1.
    # R=1 is unlikely to work for matrix multiplication.
    # But let's try.
    
    for r in [1]:
        if time.time() > deadline:
            break
        decomp = _try_find_decomposition(3, 4, 4, r, time_limit=10.0)
        if decomp is not None:
            best_rank_344 = r
            best_decomp_344 = decomp
            break
            
    R = best_rank_344
    if R < 48:
        U344 = best_decomp_344["U"]
        V344 = best_decomp_344["V"]
        W344 = best_decomp_344["W"]
        
        U8 = np.hstack([U344, U344])
        
        V8 = np.zeros((32, 2*R))
        for j in range(4):
            for k in range(4):
                V8[j*8+k, :R] = V344[j*4+k, :]
                V8[j*8+(k+4), R:] = V344[j*4+k, :]
                
        W8 = np.zeros((24, 2*R))
        for k in range(4):
            for i in range(3):
                W8[k*3+i, :R] = W344[k*3+i, :]
                W8[(k+4)*3+i, R:] = W344[k*3+i, :]
                
        if _verify(U8, V8, W8, 3, 4, 8):
            return {"U": U8, "V": V8, "W": W8}
            
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
