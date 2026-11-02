"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]
_EARLY_DEADLINE = 145  # seconds from start, leaves 5s safety margin

def _precompute_structures():
    """Precompute all index mappings and matrix structures for fast ALS."""
    n, m, p = DIMENSIONS
    
    # Target indices: T[i*m+j, j*p+k, k*n+i] = 1
    target_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_idx.append((i*m+j, j*p+k, k*n+i))
    
    # For U update at index a: need V[b,:] * W[c,:] for all (b,c) pairs
    # a = i*m+j, and for each k: b = j*p+k, c = k*n+i
    U_struct = []
    for a in range(n*m):
        i = a // m
        j = a % m
        rows = []
        for k in range(p):
            b = j * p + k
            c = k * n + i
            rows.append((b, c))
        U_struct.append(rows)
    
    # For V update at index b: need U[a,:] * W[c,:] for all (a,c) pairs
    # b = j*p+k, and for each i: a = i*m+j, and for each kk: c = kk*n+i
    V_struct = []
    for b in range(m*p):
        j = b // p
        k = b % p
        rows = []
        for i in range(n):
            a = i * m + j
            for kk in range(p):
                c = kk * n + i
                rows.append((a, c))
        V_struct.append(rows)
    
    # For W update at index c: need U[a,:] * V[b,:] for all (a,b) pairs
    # c = k*n+i, and for each ii: a = ii*m+j for each j, b = j*p+k
    W_struct = []
    for c in range(p*n):
        k = c // n
        i = c % n
        rows = []
        for ii in range(n):
            for j in range(m):
                a = ii * m + j
                b = j * p + k
                rows.append((a, b))
        W_struct.append(rows)
    
    # Target mask
    target_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for (a, b, c) in target_idx:
        target_mask[a, b, c] = True
    
    return U_struct, V_struct, W_struct, target_idx, target_mask

_U_struct, _V_struct, _W_struct, _target_idx, _target_mask = _precompute_structures()

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    max_err = 0.0
    for (a, b, c) in _target_idx:
        err = abs(T[a,b,c] - 1.0)
        if err > max_err:
            max_err = err
    
    non_target_err = np.max(np.abs(T[~_target_mask])) if (~_target_mask).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for scale in [2, 1, 4, 8]:
        U_r = np.round(U_arr * scale) / scale
        V_r = np.round(V_arr * scale) / scale
        W_r = np.round(W_arr * scale) / scale
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), tol=tol):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    Split B into 4 blocks of 3x2, use Strassen(7) + 4 rank-1 per block.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))
    V = np.zeros((m*p, 44))
    W = np.zeros((p*n, 44))
    
    col_offset = 0
    
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen rank-7 for 2x2 * 2x2 (A cols 0,1 times B rows 0,1)
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1; U[4, t] += 1
        V[k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1; U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1; W[2*k1+1, t] += -1
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1; V[8+k1, t] += -1
        W[2*k1, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1; V[k0, t] += -1
        W[2*k0, t] += 1; W[2*k0+1, t] += 1
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1; U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1; W[2*k1, t] += 1
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1; U[0, t] += -1
        V[k0, t] += 1; V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1; U[4, t] += -1
        V[8+k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1
        
        # 4 rank-1 terms for A_right * B_bottom
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

def _fast_als_decompose(rank, max_iters=200, deadline=None, seed=None, init_U=None, init_V=None, init_W=None):
    """
    Fast ALS for tensor decomposition with precomputed structures.
    """
    n, m, p = DIMENSIONS
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    if init_U is not None:
        U = np.array(init_U, dtype=float).copy()
        V = np.array(init_V, dtype=float).copy()
        W = np.array(init_W, dtype=float).copy()
    else:
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
    
    ones_p = np.ones(p)
    ones_np = np.ones(n*p)
    ones_nm = np.ones(n*m)
    
    best_err = float('inf')
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for a_idx in range(n*m):
            rows = _U_struct[a_idx]
            M = np.zeros((p, rank))
            for row, (b, c) in enumerate(rows):
                M[row, :] = V[b, :] * W[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_p, rcond=None)
                U[a_idx, :] = x
            except:
                pass
                    
        # Update V
        for b_idx in range(m*p):
            rows = _V_struct[b_idx]
            M = np.zeros((len(rows), rank))
            for row, (a, c) in enumerate(rows):
                M[row, :] = U[a, :] * W[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_np, rcond=None)
                V[b_idx, :] = x
            except:
                pass
                    
        # Update W
        for c_idx in range(p*n):
            rows = _W_struct[c_idx]
            M = np.zeros((len(rows), rank))
            for row, (a, b) in enumerate(rows):
                M[row, :] = U[a, :] * V[b, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_nm, rcond=None)
                W[c_idx, :] = x
            except:
                pass
        
        # Check convergence
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        max_err = 0.0
        for (a, b, c) in _target_idx:
            err = abs(T[a,b,c] - 1.0)
            if err > max_err:
                max_err = err
        
        non_target_err = np.max(np.abs(T[~_target_mask])) if (~_target_mask).any() else 0.0
        total_err = max_err + non_target_err
        
        if total_err < best_err:
            best_err = total_err
        
        if max_err < 1e-9 and non_target_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _polish(U, V, W, max_iters=100, tol=1e-12, deadline=None):
    """High-precision polish pass."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    ones_p = np.ones(p)
    ones_np = np.ones(n*p)
    ones_nm = np.ones(n*m)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        for a_idx in range(n*m):
            rows = _U_struct[a_idx]
            M = np.zeros((p, rank))
            for row, (b, c) in enumerate(rows):
                M[row, :] = V_arr[b, :] * W_arr[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_p, rcond=None)
                U_arr[a_idx, :] = x
            except:
                pass
                    
        for b_idx in range(m*p):
            rows = _V_struct[b_idx]
            M = np.zeros((len(rows), rank))
            for row, (a, c) in enumerate(rows):
                M[row, :] = U_arr[a, :] * W_arr[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_np, rcond=None)
                V_arr[b_idx, :] = x
            except:
                pass
                    
        for c_idx in range(p*n):
            rows = _W_struct[c_idx]
            M = np.zeros((len(rows), rank))
            for row, (a, b) in enumerate(rows):
                M[row, :] = U_arr[a, :] * V_arr[b, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_nm, rcond=None)
                W_arr[c_idx, :] = x
            except:
                pass
        
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
            
        max_err = 0.0
        for (a, b, c) in _target_idx:
            err = abs(T[a,b,c] - 1.0)
            if err > max_err:
                max_err = err
        non_target_err = np.max(np.abs(T[~_target_mask])) if (~_target_mask).any() else 0.0
        
        if max_err < tol and non_target_err < tol:
            return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
            
    return None

def _try_rank_with_warm_start(rank, deadline, base_U, base_V, base_W, num_trials=5):
    """Try to find a valid decomposition at given rank, using warm starts from higher-rank solution."""
    n, m, p = DIMENSIONS
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        if trial == 0 and base_U is not None:
            # Warm start: take first `rank` columns of the base solution
            init_U = base_U[:, :rank]
            init_V = base_V[:, :rank]
            init_W = base_W[:, :rank]
            result = _fast_als_decompose(rank, max_iters=150, deadline=deadline, seed=trial*1000+rank, 
                                          init_U=init_U, init_V=init_V, init_W=init_W)
        else:
            result = _fast_als_decompose(rank, max_iters=150, deadline=deadline, seed=trial*1000+rank)
        
        if result:
            U, V, W = result
            if _verify(U, V, W, tol=1e-6):
                polished = _polish(U, V, W, max_iters=60, tol=1e-12, deadline=deadline)
                if polished:
                    U, V, W = polished
                rounded = _round_and_verify(U, V, W)
                if rounded:
                    return rounded
                else:
                    return (U, V, W)
    
    return None

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
    deadline = start_time + _EARLY_DEADLINE
    
    # First, try the explicit rank-44 decomposition
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
    except Exception as e:
        pass
    
    # Now try to find better with ALS, using warm starts
    if best_rank == 44:
        base_U = np.array(best_U, dtype=float)
        base_V = np.array(best_V, dtype=float)
        base_W = np.array(best_W, dtype=float)
        
        for rank in range(32, 44):
            if time.time() > deadline:
                break
            
            result = _try_rank_with_warm_start(rank, deadline, base_U, base_V, base_W, num_trials=6)
            if result:
                best_U, best_V, best_W = result
                best_rank = rank
                base_U = np.array(best_U, dtype=float)
                base_V = np.array(best_V, dtype=float)
                base_W = np.array(best_W, dtype=float)
    
    return {"U": best_U, "V": best_V, "W": best_W}
