"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _precompute_indices():
    n, m, p = DIMENSIONS
    target_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_idx.append((i*m+j, j*p+k, k*n+i))
    return np.array(target_idx, dtype=int)

_TARGET_IDX = None
_NON_TARGET_MASK = None

def _get_target_idx():
    global _TARGET_IDX
    if _TARGET_IDX is None:
        _TARGET_IDX = _precompute_indices()
    return _TARGET_IDX

def _get_non_target_mask():
    global _NON_TARGET_MASK
    if _NON_TARGET_MASK is None:
        n, m, p = DIMENSIONS
        total = n*m * m*p * p*n
        flat = np.zeros(total, dtype=bool)
        ti = _get_target_idx()
        flat_idx = ti[:,0] * (m*p * p*n) + ti[:,1] * (p*n) + ti[:,2]
        flat[flat_idx] = True
        _NON_TARGET_MASK = ~flat
    return _NON_TARGET_MASK

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    target_idx = _get_target_idx()
    target_vals = T[target_idx[:,0], target_idx[:,1], target_idx[:,2]]
    max_err = np.max(np.abs(target_vals - 1.0))
    
    non_target_mask = _get_non_target_mask()
    non_target = np.abs(T.reshape(-1)[non_target_mask])
    non_target_err = np.max(non_target) if non_target.size > 0 else 0.0
    
    return max_err <= tol and non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try integers
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=tol):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    return None

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    
    Strategy: Split B (3x8) into 4 blocks of 3x2: B = [B_0, B_1, B_2, B_3]
    Each C_t = A * B_t is a 2x3 * 3x2 = 2x2 multiplication.
    
    For each 2x3*3x2 sub-problem, use rank-11 scheme:
    - Strassen rank-7 for the 2x2 output (treating A as having 3 columns)
    - Plus 4 rank-1 terms to handle the extra dimension
    
    Actually, let me think more carefully. For M_{2,3,2}:
    A is 2x3, B is 3x2, C is 2x2.
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    One approach: use the fact that we can write this as:
    C = A_0 * B_col0 + A_1 * B_col1 + A_2 * B_col2
    where A_t is column t of A (2x1), B_colk is column k of B (3x1).
    Wait, that's not right either.
    
    Let me use a different decomposition. For 2x3 * 3x2:
    Think of A as [A_left | A_right] where A_left is 2x2 (cols 0,1) and A_right is 2x1 (col 2).
    B as [B_top; B_bottom] where B_top is 2x2 (rows 0,1) and B_bottom is 1x2 (row 2).
    
    C = A_left * B_top + A_right * B_bottom
    
    A_left * B_top is 2x2 * 2x2, which has rank 7 (Strassen).
    A_right * B_bottom is 2x1 * 1x2, which has rank 4 (outer product, each of 4 entries).
    But we can share structure...
    
    Actually, A_right * B_bottom: A_right is 2x1, B_bottom is 1x2.
    (A_right * B_bottom)[i,k] = A_right[i] * B_bottom[k]
    This is a rank-1 matrix in the output, but as a tensor term it contributes to all 4 output entries.
    We need 4 separate rank-1 tensor terms for this? No:
    
    For the tensor M_{2,3,2}, the "A_right * B_bottom" part:
    T[a,b,d] = A[1*3+2, t] * B[2*2+k, t] * W[d,t] where a=i*3+2, b=2*2+k=4+k, d=k*2+i
    
    This is: for each (i,k), T[i*3+2, 4+k, k*2+i] = A_right[i] * B_bottom[k]
    We can decompose this as: for each i in {0,1}, k in {0,1}:
    term with U[i*3+2, t]=1, V[4+k, t]=1, W[k*2+i, t]=1
    That's 4 rank-1 terms.
    
    And A_left * B_top is the standard 2x2*2x2 Strassen with rank 7.
    Total: 7 + 4 = 11 per sub-block.
    4 sub-blocks: 4 * 11 = 44.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))  # 6 x 44
    V = np.zeros((m*p, 44))  # 24 x 44
    W = np.zeros((p*n, 44))  # 16 x 44
    
    col_offset = 0
    
    for block in range(4):
        # Block handles columns 2*block and 2*block+1 of B
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen rank-7 for the 2x2 * 2x2 part (A cols 0,1 times B rows 0,1)
        # A_left: rows 0,1 of A, cols 0,1 -> indices i*3+j for i in {0,1}, j in {0,1}
        # B_top: rows 0,1 of B, cols k0,k1 -> indices j*8+k for j in {0,1}, k in {k0,k1}
        # C: 2x2 output, entries C[i, k0] and C[i, k1] for i in {0,1}
        # W index: k*n+i = k*2+i
        
        # Strassen products for 2x2*2x2:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # a00 = A[0,0] -> U index 0*3+0 = 0
        # a01 = A[0,1] -> U index 0*3+1 = 1
        # a10 = A[1,0] -> U index 1*3+0 = 3
        # a11 = A[1,1] -> U index 1*3+1 = 4
        
        # b00 = B[0,k0] -> V index 0*8+k0 = k0
        # b01 = B[0,k1] -> V index 0*8+k1 = k1
        # b10 = B[1,k0] -> V index 1*8+k0 = 8+k0
        # b11 = B[1,k1] -> V index 1*8+k1 = 8+k1
        
        # C00: i=0,k=k0 -> W index k0*2+0 = 2*k0
        # C01: i=0,k=k1 -> W index k1*2+0 = 2*k1
        # C10: i=1,k=k0 -> W index k0*2+1 = 2*k0+1
        # C11: i=1,k=k1 -> W index k1*2+1 = 2*k1+1
        
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1   # a00
        U[4, t] += 1   # a11
        V[k0, t] += 1  # b00
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += 1  # C00
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1   # a10
        U[4, t] += 1   # a11
        V[k0, t] += 1  # b00
        W[2*k0+1, t] += 1  # C10
        W[2*k1+1, t] += -1  # C11
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1   # a00
        V[k1, t] += 1  # b01
        V[8+k1, t] += -1  # b11
        W[2*k1, t] += 1  # C01
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1   # a11
        V[8+k0, t] += 1  # b10
        V[k0, t] += -1  # b00
        W[2*k0, t] += 1  # C00
        W[2*k0+1, t] += 1  # C10
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1   # a00
        U[1, t] += 1   # a01
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += -1  # C00
        W[2*k1, t] += 1  # C01
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1   # a10
        U[0, t] += -1  # a00
        V[k0, t] += 1  # b00
        V[k1, t] += 1  # b01
        W[2*k1+1, t] += 1  # C11
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1   # a01
        U[4, t] += -1  # a11
        V[8+k0, t] += 1  # b10
        V[8+k1, t] += 1  # b11
        W[2*k0, t] += 1  # C00
        col_offset += 1
        
        # Now the 4 rank-1 terms for A_right * B_bottom
        # A_right = A[:, 2] -> U indices 0*3+2=2 and 1*3+2=5
        # B_bottom = B[2, :] -> V indices 2*8+k = 16+k for k in {k0, k1}
        
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1   # A[ii, 2]
                V[16+k_val, t] = 1  # B[2, k_val]
                W[k_val*2+ii, t] = 1  # C[ii, k_val]
                col_offset += 1
    
    assert col_offset == 44, f"Expected 44 columns, got {col_offset}"
    
    return U.tolist(), V.tolist(), W.tolist()

def _als_decompose(rank, max_iters=100, deadline=None, seed=None):
    """
    Fast Alternating Least Squares for tensor decomposition using vectorized ops.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    target_idx = _get_target_idx()
    non_target_mask = _get_non_target_mask()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U: for each a = i*m+j, solve M_a x = ones(p)
        # M_a[k, :] = V[j*p+k, :] * W[k*n+i, :]
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                try:
                    U[a, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
                    
        # Update V: for each b = j*p+k, solve M_b x = ones(n*p)
        # M_b[idx, :] = U[i*m+j, :] * W[kk*n+i, :] for all (i, kk)
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.array([U[i*m+j, :] * W[kk*n+i, :] for i in range(n) for kk in range(p)])
                try:
                    V[b, :] = np.linalg.lstsq(M, np.ones(n*p), rcond=None)[0]
                except:
                    pass
                    
        # Update W: for each c = k*n+i, solve M_c x = ones(n*m)
        # M_c[idx, :] = U[ii*m+j, :] * V[j*p+k, :] for all (ii, j)
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.array([U[ii*m+j, :] * V[j*p+k, :] for ii in range(n) for j in range(m)])
                try:
                    W[c, :] = np.linalg.lstsq(M, np.ones(n*m), rcond=None)[0]
                except:
                    pass
                    
        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target_vals = T[target_idx[:,0], target_idx[:,1], target_idx[:,2]]
        max_err = np.max(np.abs(target_vals - 1.0))
        non_target = np.abs(T.reshape(-1)[non_target_mask])
        non_target_err = np.max(non_target) if non_target.size > 0 else 0.0
        
        if max_err < 1e-9 and non_target_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _als_polish(U, V, W, max_iters=50, deadline=None, tol=1e-9):
    """
    Fast ALS polish on existing U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    target_idx = _get_target_idx()
    non_target_mask = _get_non_target_mask()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                try:
                    U[a, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.array([U[i*m+j, :] * W[kk*n+i, :] for i in range(n) for kk in range(p)])
                try:
                    V[b, :] = np.linalg.lstsq(M, np.ones(n*p), rcond=None)[0]
                except:
                    pass
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.array([U[ii*m+j, :] * V[j*p+k, :] for ii in range(n) for j in range(m)])
                try:
                    W[c, :] = np.linalg.lstsq(M, np.ones(n*m), rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target_vals = T[target_idx[:,0], target_idx[:,1], target_idx[:,2]]
        max_err = np.max(np.abs(target_vals - 1.0))
        non_target = np.abs(T.reshape(-1)[non_target_mask])
        non_target_err = np.max(non_target) if non_target.size > 0 else 0.0
        
        if max_err < tol and non_target_err < tol:
            return True
    return False

def _try_reduce_rank(U0, V0, W0, target_rank, max_iters=50, deadline=None, seed=None):
    """
    Try to reduce rank from len(U0) to target_rank by dropping terms and re-optimizing.
    Start from the rank-44 solution, zero out some columns, then ALS polish.
    Uses structured removal: prefer removing entire rank-1 blocks (the 4 outer-product terms per block).
    """
    n, m, p = DIMENSIONS
    cur_rank = len(U0[0])
    rng = np.random.RandomState(seed) if seed is not None else np.random
    U = np.array(U0, dtype=float).copy()
    V = np.array(V0, dtype=float).copy()
    W = np.array(W0, dtype=float).copy()
    
    n_drop = cur_rank - target_rank
    
    # Structured removal: the rank-44 has 4 blocks of 11 terms each.
    # Each block: 7 Strassen + 4 outer-product rank-1 terms.
    # The 4 outer-product terms are at positions 7-10 within each block of 11.
    # Prefer removing outer-product terms first (they're more "redundant").
    outer_prod_cols = []
    for block in range(4):
        base = block * 11
        for offset in range(4):
            outer_prod_cols.append(base + 7 + offset)
    
    # Try removing outer-product columns first, then random
    if n_drop <= len(outer_prod_cols):
        cols_to_zero = rng.choice(outer_prod_cols, n_drop, replace=False)
    else:
        # Need to remove more than just outer-product terms
        remaining = [c for c in range(cur_rank) if c not in set(outer_prod_cols)]
        need_more = n_drop - len(outer_prod_cols)
        cols_to_zero = list(outer_prod_cols)
        if need_more > 0:
            extra = rng.choice(remaining, min(need_more, len(remaining)), replace=False)
            cols_to_zero.extend(extra.tolist())
    
    cols_to_zero = set(cols_to_zero.tolist())
    
    # Keep only the target_rank columns
    keep_cols = [c for c in range(cur_rank) if c not in cols_to_zero]
    U = U[:, keep_cols]
    V = V[:, keep_cols]
    W = W[:, keep_cols]
    
    # ALS polish with more iterations for better convergence
    success = _als_polish(U, V, W, max_iters=max_iters, deadline=deadline, tol=1e-8)
    
    if success:
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
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # First, try the explicit rank-44 decomposition
    U44 = V44 = W44 = None
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
    
    # Try to reduce from 44 to lower ranks using structured term-dropping + ALS polish
    if best_rank == 44 and U44 is not None and time.time() < deadline:
        for target_rank in [43, 42, 41, 40, 39, 38, 37, 36]:
            if time.time() > deadline - 3:
                break
            found = False
            # More trials for smaller drops, fewer for larger
            if target_rank >= 41:
                max_trials = 25
            elif target_rank >= 39:
                max_trials = 20
            else:
                max_trials = 12
            for trial in range(max_trials):
                if time.time() > deadline - 3:
                    break
                seed = trial * 7 + 13 + target_rank
                result = _try_reduce_rank(U44, V44, W44, target_rank, 
                                         max_iters=40, deadline=deadline, seed=seed)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                        else:
                            best_U, best_V, best_W = U, V, W
                        best_rank = target_rank
                        found = True
                        break
            if not found:
                # If we can't do target_rank, try next lower
                continue
    
    # Fallback: ALS from scratch for ranks below best
    if time.time() < deadline - 10:
        for rank in range(max(30, best_rank - 10), best_rank):
            if time.time() > deadline - 10:
                break
            for trial in range(8):
                if time.time() > deadline - 10:
                    break
                seed = rank * 100 + trial * 37
                result = _als_decompose(rank, max_iters=60, deadline=deadline, seed=seed)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                        else:
                            best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        break
            if best_rank <= 40:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
