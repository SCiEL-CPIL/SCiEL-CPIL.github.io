"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

_DEADLINE = 150.0
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _deadline_reached(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: A00+A11, B00+B11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # M2: A10+A11, B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # M3: A00, B01-B11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # M4: A11, B10-B00
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # M5: A00+A01, B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # M6: A10-A00, B00+B01
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # M7: A01-A11, B10+B11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[k,i]
    # C00 (0,0): M1 - M4 + M6 + M7
    W[0, 0] = 1; W[0, 3] = -1; W[0, 5] = 1; W[0, 6] = 1
    # C01 (0,1): M5 + M4
    W[1, 4] = 1; W[1, 3] = 1
    # C10 (1,0): M1 + M2
    W[2, 0] = 1; W[2, 1] = 1
    # C11 (1,1): M3 - M1 + M5 + M7
    W[3, 2] = 1; W[3, 0] = -1; W[3, 4] = 1; W[3, 6] = 1
    
    return U, V, W

def _trivial_2x2x1():
    """Return the rank-4 trivial decomposition for 2x2 * 2x1 -> 2x1."""
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((2, rank))
    W = np.zeros((2, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _build_rank32():
    """Build the rank-32 block decomposition."""
    n, m, p = 2, 2, 9
    num_full = p // 2  # 4
    remainder = p % 2   # 1
    rank = num_full * 7 + remainder * 4
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_s, V_s, W_s = _strassen_2x2()
    U_t, V_t, W_t = _trivial_2x2x1()
    
    col_offset = 0
    for block in range(num_full):
        U[:, col_offset:col_offset+7] = U_s
        for j in range(m):
            for k_local in range(2):
                k_global = 2 * block + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_s[j, :]
        for k_local in range(2):
            for i in range(n):
                k_global = 2 * block + k_local
                W[k_global * n + i, col_offset:col_offset+7] = W_s[k_local * n + i, :]
        col_offset += 7
    
    if remainder > 0:
        k_start = 2 * num_full
        U[:, col_offset:col_offset+4] = U_t
        for j in range(m):
            V[j * p + k_start, col_offset:col_offset+4] = V_t[j, :]
        for i in range(n):
            W[k_start * n + i, col_offset:col_offset+4] = W_t[i, :]
        col_offset += 4
        
    return U, V, W

def _compute_tensor(U, V, W, n, m, p):
    """Compute the tensor from the decomposition."""
    tensor = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                tensor[i, j, k] = s
    return tensor

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m:
        return False
    if V.shape[0] != m*p:
        return False
    if W.shape[0] != p*n:
        return False
        
    tensor = _compute_tensor(U, V, W, n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(tensor[i, j, k] - 1.0) > tol:
                    return False
    return True

def _to_list(mat):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in mat:
        r = []
        for val in row:
            v = round(val * 2) / 2.0
            if abs(v - round(v)) < 1e-12:
                v = int(round(v))
            r.append(v)
        result.append(r)
    return result

def _try_reduce_rank(U, V, W, target_rank, n, m, p):
    """Try to reduce the rank by optimizing a subset of terms."""
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    # We will keep the first 'target_rank' terms and try to adjust them
    # to fit the tensor, effectively dropping the last 'current_rank - target_rank' terms.
    # This is a heuristic.
    
    U0 = U[:, :target_rank].copy()
    V0 = V[:, :target_rank].copy()
    W0 = W[:, :target_rank].copy()
    
    # Reshape for optimization
    # Variables: U0, V0, W0 flattened
    # Objective: sum of squares of (Target - Decomposed)
    
    def objective(params):
        u_flat, v_flat, w_flat = np.split(params, [U0.size, U0.size + V0.size])
        u = u_flat.reshape(U0.shape)
        v = v_flat.reshape(V0.shape)
        w = w_flat.reshape(W0.shape)
        
        tensor = _compute_tensor(u, v, w, n, m, p)
        error = tensor - 1.0
        return error.flatten()
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    try:
        # Use least squares to minimize the error
        result = least_squares(objective, x0, method='lm', max_nfev=1000)
        
        if result.success or np.linalg.norm(result.fun) < 1e-8:
            u, v, w = np.split(result.x, [U0.size, U0.size + V0.size])
            u = u.reshape(U0.shape)
            v = v.reshape(V0.shape)
            w = w.reshape(W0.shape)
            
            if _verify(u, v, w, n, m, p):
                return u, v, w
    except Exception:
        pass
        
    return None

def construct():
    _START_TIME = time.time()
    n, m, p = DIMENSIONS
    
    # Start with rank 32
    U, V, W = _build_rank32()
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try to reduce to rank 31
    if not _deadline_reached(20.0):
        res = _try_reduce_rank(U, V, W, 31, n, m, p)
        if res is not None:
            best_U, best_V, best_W = res
            best_rank = 31
            
    # Try to reduce to rank 30 if we have time
    if best_rank == 31 and not _deadline_reached(10.0):
         # We might need to start from the rank 31 solution
         res = _try_reduce_rank(best_U, best_V, best_W, 30, n, m, p)
         if res is not None:
             best_U, best_V, best_W = res
             best_rank = 30

    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}
# EVOLVE-BLOCK-END
