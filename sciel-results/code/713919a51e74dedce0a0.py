"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]
_N, _M, _P = 2, 3, 8
_NMP = _N * _M * _P  # 48

# Precompute target indices and mask
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
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute T only at target positions using einsum
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We only need to check target positions and that non-target are ~0
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

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))  # 6 x 44
    V = np.zeros((m*p, 44))  # 24 x 44
    W = np.zeros((p*n, 44))  # 16 x 44
    
    col_offset = 0
    
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1
        W[2*k1+1, t] += -1
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1
        V[8+k1, t] += -1
        W[2*k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1
        V[k0, t] += -1
        W[2*k0, t] += 1
        W[2*k0+1, t] += 1
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1
        U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1
        W[2*k1, t] += 1
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1
        U[0, t] += -1
        V[k0, t] += 1
        V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1
        U[4, t] += -1
        V[8+k0, t] += 1
        V[8+k1, t] += 1
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

def _precompute_indices():
    """Precompute index mappings for faster ALS iterations."""
    n, m, p = DIMENSIONS
    U_indices = []
    for i in range(n):
        for j in range(m):
            entries = []
            for k in range(p):
                b = j * p + k
                c = k * n + i
                entries.append((k, b, c))
            U_indices.append(entries)
    
    V_indices = []
    for j in range(m):
        for k in range(p):
            entries = []
            for i in range(n):
                for kk in range(p):
                    a = i * m + j
                    c = kk * n + i
                    entries.append((i, kk, a, c))
            V_indices.append(entries)
    
    W_indices = []
    for k in range(p):
        for i in range(n):
            entries = []
            for ii in range(n):
                for j in range(m):
                    a = ii * m + j
                    b = j * p + k
                    entries.append((ii, j, a, b))
            W_indices.append(entries)
    
    return U_indices, V_indices, W_indices

_U_indices, _V_indices, _W_indices = _precompute_indices()

def _compute_residual(U, V, W):
    """Compute residual = ||T - target||_F^2."""
    T = np.einsum('ar,br,cr->abc', U, V, W)
    # Target is 1 at _TARGET_IDX positions, 0 elsewhere
    residual = T.copy()
    for a, b, c in _TARGET_IDX:
        residual[a, b, c] -= 1.0
    return np.sum(residual**2)

def _als_decompose(rank, max_iters=200, deadline=None, seed=None, init=None):
    """
    Vectorized Alternating Least Squares for tensor decomposition.
    """
    n, m, p = DIMENSIONS
    
    if init is not None:
        U = np.array(init[0], dtype=float).copy()
        V = np.array(init[1], dtype=float).copy()
        W = np.array(init[2], dtype=float).copy()
    else:
        if seed is not None:
            np.random.seed(seed)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
    
    best_residual = np.inf
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U: for each a, solve (V[b,:] * W[c,:]) @ x = ones(p)
        # where b = j*p+k, c = k*n+i for (i,j,k) with a = i*m+j
        for a_idx in range(n*m):
            i = a_idx // m
            j = a_idx % m
            # For each k in 0..p-1: b = j*p+k, c = k*n+i
            # M[k, :] = V[j*p+k, :] * W[k*n+i, :]
            M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                U[a_idx, :] = x
            except:
                pass
                    
        # Update V: for each b, solve (U[a,:] * W[c,:]) @ x = ones(n*p)
        # where b = j*p+k, and for each (i,kk): a = i*m+j, c = kk*n+i
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
                    
        # Update W: for each c, solve (U[a,:] * V[b,:]) @ x = ones(n*m)
        # where c = k*n+i, and for each (ii,j): a = ii*m+j, b = j*p+k
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
        for (a, b, c) in _TARGET_IDX:
            err = abs(T[a,b,c] - 1.0)
            if err > max_err:
                max_err = err
        
        non_target_err = np.max(np.abs(T[~_TARGET_MASK]))
        
        if max_err < 1e-9 and non_target_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
        
        residual = _compute_residual(U, V, W)
        if residual < best_residual:
            best_residual = residual
    
    return None

def _polish(U, V, W, max_iters=100, tol=1e-14, deadline=None):
    """High-precision polish pass (vectorized)."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        for a_idx in range(n*m):
            i = a_idx // m
            j = a_idx % m
            M = np.array([V_arr[j*p+k, :] * W_arr[k*n+i, :] for k in range(p)])
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                U_arr[a_idx, :] = x
            except:
                pass
                    
        for b_idx in range(m*p):
            j = b_idx // p
            k = b_idx % p
            rows = [U_arr[i*m+j, :] * W_arr[kk*n+i, :] for i in range(n) for kk in range(p)]
            M = np.array(rows)
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(rows)), rcond=None)
                V_arr[b_idx, :] = x
            except:
                pass
                    
        for c_idx in range(p*n):
            k = c_idx // n
            i = c_idx % n
            rows = [U_arr[ii*m+j, :] * V_arr[j*p+k, :] for ii in range(n) for j in range(m)]
            M = np.array(rows)
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(rows)), rcond=None)
                W_arr[c_idx, :] = x
            except:
                pass
        
        T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
            
        max_err = 0.0
        for (a, b, c) in _TARGET_IDX:
            err = abs(T[a,b,c] - 1.0)
            if err > max_err:
                max_err = err
        non_target_err = np.max(np.abs(T[~_TARGET_MASK]))
        
        if max_err < tol and non_target_err < tol:
            return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
            
    return None

def _randomized_projection_search(rank, deadline, seed=0):
    """
    Randomized rank-reduction: start from a higher-rank valid decomposition,
    then try to project onto lower-rank subspaces via random linear combinations.
    """
    n, m, p = DIMENSIONS
    
    # Start from rank-44 decomposition
    U44, V44, W44 = _build_rank44_decomposition()
    U44_arr = np.array(U44, dtype=float)
    V44_arr = np.array(V44, dtype=float)
    W44_arr = np.array(W44, dtype=float)
    
    rng = np.random.RandomState(seed)
    
    # Try to reduce from 44 down to target rank
    current_rank = 44
    U_cur = U44_arr[:, :current_rank].copy()
    V_cur = V44_arr[:, :current_rank].copy()
    W_cur = W44_arr[:, :current_rank].copy()
    
    while current_rank > rank:
        if deadline and time.time() > deadline:
            break
        
        # Try to merge two columns into one via random projection
        # Pick two random columns to merge
        idx1 = rng.randint(0, current_rank)
        idx2 = rng.randint(0, current_rank)
        if idx1 == idx2:
            continue
        
        # Create a new column that is a linear combination
        alpha = rng.uniform(-2, 2)
        beta = rng.uniform(-2, 2)
        
        U_new = np.zeros((n*m, current_rank - 1))
        V_new = np.zeros((m*p, current_rank - 1))
        W_new = np.zeros((p*n, current_rank - 1))
        
        col = 0
        for i in range(current_rank):
            if i == idx1:
                U_new[:, col] = alpha * U_cur[:, idx1] + beta * U_cur[:, idx2]
                V_new[:, col] = alpha * V_cur[:, idx1] + beta * V_cur[:, idx2]
                W_new[:, col] = alpha * W_cur[:, idx1] + beta * W_cur[:, idx2]
                col += 1
            elif i == idx2:
                continue
            else:
                U_new[:, col] = U_cur[:, i]
                V_new[:, col] = V_cur[:, i]
                W_new[:, col] = W_cur[:, i]
                col += 1
        
        # Verify if this projection still works
        if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), tol=1e-8):
            current_rank -= 1
            U_cur = U_new
            V_cur = V_new
            W_cur = W_new
        else:
            # Try a few more random combinations before giving up on this merge
            success = False
            for attempt in range(20):
                alpha = rng.uniform(-3, 3)
                beta = rng.uniform(-3, 3)
                
                U_new = np.zeros((n*m, current_rank - 1))
                V_new = np.zeros((m*p, current_rank - 1))
                W_new = np.zeros((p*n, current_rank - 1))
                
                col = 0
                for i in range(current_rank):
                    if i == idx1:
                        U_new[:, col] = alpha * U_cur[:, idx1] + beta * U_cur[:, idx2]
                        V_new[:, col] = alpha * V_cur[:, idx1] + beta * V_cur[:, idx2]
                        W_new[:, col] = alpha * W_cur[:, idx1] + beta * W_cur[:, idx2]
                        col += 1
                    elif i == idx2:
                        continue
                    else:
                        U_new[:, col] = U_cur[:, i]
                        V_new[:, col] = V_cur[:, i]
                        W_new[:, col] = W_cur[:, i]
                        col += 1
                
                if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), tol=1e-8):
                    current_rank -= 1
                    U_cur = U_new
                    V_cur = V_new
                    W_cur = W_new
                    success = True
                    break
            
            if not success:
                break
    
    if current_rank == rank:
        return U_cur.tolist(), V_cur.tolist(), W_cur.tolist()
    
    return None

def _gf5_rank_search(rank, deadline, seed=0):
    """
    Search for rank-r decomposition over GF(5), then lift to integers.
    """
    n, m, p = DIMENSIONS
    MOD = 5
    
    rng = np.random.RandomState(seed)
    
    # Build target tensor over GF(5)
    T_mod = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_mod[i*m+j, j*p+k, k*n+i] = 1
    
    # Try random decompositions over GF(5)
    for trial in range(200):
        if deadline and time.time() > deadline:
            break
        
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        # Initialize randomly
        for r in range(rank):
            U[:, r] = rng.randint(0, MOD, size=n*m) % MOD
            V[:, r] = rng.randint(0, MOD, size=m*p) % MOD
            W[:, r] = rng.randint(0, MOD, size=p*n) % MOD
        
        # ALS over GF(5) - simplified: just check if random works
        # This is a very rough approach; better to use structured search
        
        # Check
        T_check = np.zeros((n*m, m*p, p*n), dtype=int)
        for r in range(rank):
            T_check += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        T_check %= MOD
        
        if np.array_equal(T_check, T_mod):
            # Found a valid decomposition over GF(5)
            # Lift to integers (values are already 0-4)
            return U.tolist(), V.tolist(), W.tolist()
    
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
    deadline = start_time + 145  # Safety margin for 150s kill
    
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
    except Exception:
        pass
    
    # Try ALS from the rank-44 decomposition as initialization, targeting lower ranks
    if best_rank == 44 and time.time() < deadline:
        for rank in range(40, 44):
            if time.time() > deadline - 5:
                break
            # Use rank-44 as init, zero out extra columns
            U_init = np.array(U44, dtype=float)[:, :rank]
            V_init = np.array(V44, dtype=float)[:, :rank]
            W_init = np.array(W44, dtype=float)[:, :rank]
            
            result = _als_decompose(rank, max_iters=50, deadline=deadline, init=(U_init.tolist(), V_init.tolist(), W_init.tolist()))
            if result:
                U, V, W = result
                if _verify(U, V, W, tol=1e-6):
                    polished = _polish(U, V, W, max_iters=20, tol=1e-14, deadline=deadline)
                    if polished:
                        U, V, W = polished
                    rounded = _round_and_verify(U, V, W)
                    if rounded:
                        best_U, best_V, best_W = rounded
                    else:
                        best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break
    
    # Try random ALS for lower ranks
    if best_rank > 36 and time.time() < deadline:
        for rank in range(34, min(best_rank, 40)):
            if time.time() > deadline - 10:
                break
            for trial in range(12):
                if time.time() > deadline - 5:
                    break
                result = _als_decompose(rank, max_iters=60, deadline=deadline, seed=trial*1000+rank)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        polished = _polish(U, V, W, max_iters=30, tol=1e-14, deadline=deadline)
                        if polished:
                            U, V, W = polished
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                        else:
                            best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        break
            if best_rank < 44:
                break
    
    # Final polish if we have a non-schoolbook solution
    if best_rank < schoolbook_rank and time.time() < deadline - 2:
        polished = _polish(best_U, best_V, best_W, max_iters=20, tol=1e-14, deadline=deadline)
        if polished:
            best_U, best_V, best_W = polished
            rounded = _round_and_verify(best_U, best_V, best_W)
            if rounded:
                best_U, best_V, best_W = rounded
    
    return {"U": best_U, "V": best_V, "W": best_W}
