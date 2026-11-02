import time
import numpy as np

DIMENSIONS = [2, 3, 8]
_N, _M, _P = 2, 3, 8
_NMP = _N * _M * _P  # 48

# Precompute target indices and mask for full problem
_TARGET_IDX = []
for _i in range(_N):
    for _j in range(_M):
        for _k in range(_P):
            _TARGET_IDX.append((_i * _M + _j, _j * _P + _k, _k * _N + _i))

_TARGET_MASK = np.zeros((_N*_M, _M*_P, _P*_N), dtype=bool)
for _a, _b, _c in _TARGET_IDX:
    _TARGET_MASK[_a, _b, _c] = True

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    max_err = 0.0
    for a, b, c in _TARGET_IDX:
        err = abs(T[a,b,c] - 1.0)
        if err > max_err:
            max_err = err
        if max_err > tol:
            return False
    
    non_target_err = np.max(np.abs(T[~_TARGET_MASK]))
    return non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try integers
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=tol):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
    return None

def _als_decompose_small(n, m, p, rank, max_iters=200, deadline=None, seed=None):
    """
    ALS for small tensor decomposition.
    """
    if seed is not None:
        np.random.seed(seed)
        
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Precompute target indices for this small problem
    target_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_idx.append((i*m + j, j*p + k, k*n + i))
                
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for a, b, c in target_idx:
        mask[a, b, c] = True
        
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for a_idx in range(n*m):
            i = a_idx // m
            j = a_idx % m
            M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                U[a_idx, :] = x
            except:
                pass
                    
        # Update V
        for b_idx in range(m*p):
            j = b_idx // p
            k = b_idx % p
            rows = []
            for i in range(n):
                for kk in range(p):
                    a = i * m + j
                    c = kk * n + i
                    rows.append(U[a, :] * W[c, :])
            M = np.array(rows)
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(rows)), rcond=None)
                V[b_idx, :] = x
            except:
                pass
                    
        # Update W
        for c_idx in range(p*n):
            k = c_idx // n
            i = c_idx % n
            rows = []
            for ii in range(n):
                for j in range(m):
                    a = ii * m + j
                    b = j * p + k
                    rows.append(U[a, :] * V[b, :])
            M = np.array(rows)
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(rows)), rcond=None)
                W[c_idx, :] = x
            except:
                pass
                    
        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
        max_err = 0.0
        for (a, b, c) in target_idx:
            err = abs(T[a,b,c] - 1.0)
            if err > max_err:
                max_err = err
        non_target_err = np.max(np.abs(T[~mask]))
        
        if max_err < 1e-9 and non_target_err < 1e-9:
            return U, V, W
            
    return None

def _find_rank10_232(deadline):
    """
    Find rank-10 decomposition for 2x3x2.
    """
    n, m, p = 2, 3, 2
    rank = 10
    
    # Try multiple seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        res = _als_decompose_small(n, m, p, rank, max_iters=100, deadline=deadline, seed=seed)
        if res:
            U, V, W = res
            # Try to round
            U_arr, V_arr, W_arr = np.array(U), np.array(V), np.array(W)
            # Check integer/half-integer
            for scale in [1, 2]:
                U_t = np.round(U_arr * scale) / scale
                V_t = np.round(V_arr * scale) / scale
                W_t = np.round(W_arr * scale) / scale
                # Verify rounded version
                T = np.einsum('ar,br,cr->abc', U_t, V_t, W_t)
                target_idx = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            target_idx.append((i*m + j, j*p + k, k*n + i))
                mask = np.zeros((n*m, m*p, p*n), dtype=bool)
                for a, b, c in target_idx:
                    mask[a, b, c] = True
                max_err = 0.0
                for a, b, c in target_idx:
                    err = abs(T[a,b,c] - 1.0)
                    if err > max_err: max_err = err
                non_target_err = np.max(np.abs(T[~mask]))
                if max_err < 1e-6 and non_target_err < 1e-6:
                    return U_t.tolist(), V_t.tolist(), W_t.tolist()
    return None

def _build_rank40_decomposition():
    """
    Build explicit rank-40 decomposition for 2x3x8 using rank-10 blocks.
    """
    n, m, p = 2, 3, 8
    rank = 40
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Find one rank-10 block
    deadline = time.time() + 10
    block = _find_rank10_232(deadline)
    
    if block is None:
        return None
        
    U_b, V_b, W_b = block
    U_b = np.array(U_b)
    V_b = np.array(V_b)
    W_b = np.array(W_b)
    
    # Assemble 4 blocks
    for b in range(4):
        offset = b * 10
        # U is the same for all blocks (A is shared)
        U[:, offset:offset+10] = U_b
        
        # V: index is j*p + k. For block b, k is 2b, 2b+1.
        for j in range(m):
            for k_prime in range(2):
                k_val = 2*b + k_prime
                V[j*p + k_val, offset:offset+10] = V_b[j, :]
                
        # W: index is k*n + i. For block b, k is 2b, 2b+1.
        for k_prime in range(2):
            k_val = 2*b + k_prime
            for i in range(n):
                W[k_val*n + i, offset:offset+10] = W_b[k_prime, :]
                
    return U.tolist(), V.tolist(), W.tolist()

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3x8.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))
    V = np.zeros((m*p, 44))
    W = np.zeros((p*n, 44))
    
    col_offset = 0
    
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        t = col_offset
        U[0, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        t = col_offset
        U[3, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1
        W[2*k1+1, t] += -1
        col_offset += 1
        
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1
        V[8+k1, t] += -1
        W[2*k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1
        V[k0, t] += -1
        W[2*k0, t] += 1
        W[2*k0+1, t] += 1
        col_offset += 1
        
        t = col_offset
        U[0, t] += 1
        U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1
        W[2*k1, t] += 1
        col_offset += 1
        
        t = col_offset
        U[3, t] += 1
        U[0, t] += -1
        V[k0, t] += 1
        V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        t = col_offset
        U[1, t] += 1
        U[4, t] += -1
        V[8+k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1
        
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1
                V[16+k_val, t] = 1
                W[k_val*2+ii, t] = 1
                col_offset += 1
    
    assert col_offset == 44, f"Expected 44 columns, got {col_offset}"
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 145
    
    # Try rank 40
    try:
        U40, V40, W40 = _build_rank40_decomposition()
        if U40:
            if _verify(U40, V40, W40, tol=1e-6):
                rounded = _round_and_verify(U40, V40, W40)
                if rounded:
                    best_U, best_V, best_W = rounded
                    best_rank = 40
                else:
                    best_U, best_V, best_W = U40, V40, W40
                    best_rank = 40
    except Exception:
        pass
    
    # Try rank 44 if 40 failed or not verified
    if best_rank > 44:
        try:
            U44, V44, W44 = _build_rank44_decomposition()
            if _verify(U44, V44, W44, tol=1e-6):
                rounded = _round_and_verify(U44, V44, W44)
                if rounded:
                    best_U, best_V, best_W = rounded
                    best_rank = 44
                else:
                    best_U, best_V, best_W = U44, V44, W44
                    best_rank = 44
        except Exception:
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
