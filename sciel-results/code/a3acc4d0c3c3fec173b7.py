"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
    
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

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

def _als_2x3x2(rank, max_iters=100, deadline=None, init=None):
    """ALS for 2x3x2 tensor decomposition."""
    n, m, p = 2, 3, 2
    if init is not None:
        U = np.array(init[0], dtype=float)
        V = np.array(init[1], dtype=float)
        W = np.array(init[2], dtype=float)
    else:
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[a, :] = x
                except:
                    pass
                    
        # Update V
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n*p, rank))
                idx = 0
                for i in range(n):
                    for kk in range(p):
                        a = i * m + j
                        c = kk * n + i
                        M[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                    V[b, :] = x
                except:
                    pass
                    
        # Update W
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((n*m, rank))
                idx = 0
                for ii in range(n):
                    for j in range(m):
                        a = ii * m + j
                        b = j * p + k
                        M[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                    W[c, :] = x
                except:
                    pass
                    
        # Check convergence
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    err = abs(T[a,b,c] - 1.0)
                    if err > max_err:
                        max_err = err
        target = np.zeros((n*m, m*p, p*n), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = True
        non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
        
        if max_err < 1e-8 and non_target_err < 1e-8:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _find_2x3x2_decomposition(rank, deadline):
    """Find decomposition for 2x3x2 with given rank."""
    for trial in range(20):
        if time.time() > deadline:
            break
        res = _als_2x3x2(rank, max_iters=50, deadline=deadline)
        if res:
            U, V, W = res
            # Verify for 2x3x2
            n, m, p = 2, 3, 2
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            T = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        err = abs(T[a,b,c] - 1.0)
                        if err > max_err:
                            max_err = err
            target = np.zeros((n*m, m*p, p*n), dtype=bool)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[i*m+j, j*p+k, k*n+i] = True
            non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
            if max_err < 1e-6 and non_target_err < 1e-6:
                return U, V, W
    return None

def _build_full_from_2x3x2(U2, V2, W2, num_blocks):
    """
    Build 2x3x(2*num_blocks) decomposition from 2x3x2 decomposition.
    """
    n, m = 2, 3
    p_block = 2
    p_total = p_block * num_blocks
    rank = len(U2[0])
    
    U_full = np.zeros((n*m, rank * num_blocks))
    V_full = np.zeros((m*p_total, rank * num_blocks))
    W_full = np.zeros((p_total*n, rank * num_blocks))
    
    for b in range(num_blocks):
        col_offset = b * rank
        U_full[:, col_offset:col_offset+rank] = np.array(U2, dtype=float)
        
        # V: block b corresponds to rows b*2 to b*2+1 in V2?
        # V2 has shape (m*p_block, rank) = (3*2, rank) = (6, rank)
        # V_full has shape (m*p_total, rank*num_blocks) = (3*8, ...) = (24, ...)
        # For block b, the rows of V_full are b*2 to b*2+1? No.
        # V_full index is j*p_total + k.
        # For a specific block b, the columns of B are k in {2b, 2b+1}.
        # The rows of B are j in {0,1,2}.
        # So V_full indices are j*8 + 2b and j*8 + 2b + 1.
        # V2 indices are j*2 + k_local where k_local in {0,1}.
        # So V2[j*2 + k_local] maps to V_full[j*8 + 2b + k_local].
        
        for j in range(m):
            for k_local in range(p_block):
                v2_idx = j * p_block + k_local
                v_full_idx = j * p_total + b * p_block + k_local
                V_full[v_full_idx, col_offset:col_offset+rank] = np.array(V2[v2_idx], dtype=float)
                
        # W: W2 has shape (p_block*n, rank) = (2*2, rank) = (4, rank)
        # W_full has shape (p_total*n, rank*num_blocks) = (8*2, ...) = (16, ...)
        # W_full index is k*n + i.
        # For block b, k in {2b, 2b+1}. i in {0,1}.
        # W2 index is k_local*n + i.
        # So W2[k_local*n + i] maps to W_full[(2b+k_local)*n + i].
        
        for k_local in range(p_block):
            for i in range(n):
                w2_idx = k_local * n + i
                w_full_idx = (b * p_block + k_local) * n + i
                W_full[w_full_idx, col_offset:col_offset+rank] = np.array(W2[w2_idx], dtype=float)
                
    return U_full.tolist(), V_full.tolist(), W_full.tolist()

def _als_decompose_full(rank, max_iters=100, deadline=None, init=None):
    """ALS for full 2x3x8 tensor decomposition."""
    n, m, p = DIMENSIONS
    if init is not None:
        U = np.array(init[0], dtype=float)
        V = np.array(init[1], dtype=float)
        W = np.array(init[2], dtype=float)
    else:
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[a, :] = x
                except:
                    pass
                    
        # Update V
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n*p, rank))
                idx = 0
                for i in range(n):
                    for kk in range(p):
                        a = i * m + j
                        c = kk * n + i
                        M[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                    V[b, :] = x
                except:
                    pass
                    
        # Update W
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((n*m, rank))
                idx = 0
                for ii in range(n):
                    for j in range(m):
                        a = ii * m + j
                        b = j * p + k
                        M[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                    W[c, :] = x
                except:
                    pass
                    
        # Check convergence
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    err = abs(T[a,b,c] - 1.0)
                    if err > max_err:
                        max_err = err
        target = np.zeros((n*m, m*p, p*n), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = True
        non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
        
        if max_err < 1e-9 and non_target_err < 1e-9:
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
    deadline = start_time + 140
    
    # Strategy: Find 2x3x2 decomposition and replicate
    # Try rank 8 for 2x3x2
    U2, V2, W2 = _find_2x3x2_decomposition(8, deadline)
    
    if U2:
        U_full, V_full, W_full = _build_full_from_2x3x2(U2, V2, W2, 4)
        if _verify(U_full, V_full, W_full, tol=1e-6):
            best_U, best_V, best_W = U_full, V_full, W_full
            best_rank = 32
            
            # Try to reduce rank further from 32
            for target_rank in [31, 30, 29, 28]:
                if time.time() > deadline:
                    break
                for trial in range(5):
                    if time.time() > deadline:
                        break
                    # Initialize with random linear combinations of the 32 factors
                    A_rand = np.random.randn(target_rank, 32)
                    B_rand = np.random.randn(target_rank, 32)
                    C_rand = np.random.randn(target_rank, 32)
                    U_init = np.array(best_U, dtype=float) @ A_rand.T
                    V_init = np.array(best_V, dtype=float) @ B_rand.T
                    W_init = np.array(best_W, dtype=float) @ C_rand.T
                    
                    result = _als_decompose_full(target_rank, max_iters=50, deadline=deadline, init=(U_init.tolist(), V_init.tolist(), W_init.tolist()))
                    if result:
                        if _verify(result[0], result[1], result[2], tol=1e-6):
                            best_U, best_V, best_W = result
                            best_rank = target_rank
                            break
                if best_rank == target_rank:
                    break
    else:
        # Fallback: Try rank 9 for 2x3x2
        U2, V2, W2 = _find_2x3x2_decomposition(9, deadline)
        if U2:
            U_full, V_full, W_full = _build_full_from_2x3x2(U2, V2, W2, 4)
            if _verify(U_full, V_full, W_full, tol=1e-6):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = 36

    # If we didn't find anything better than 48, try generic ALS on full tensor for rank 40-44
    if best_rank > 40:
        for target_rank in [44, 43, 42, 41, 40]:
            if time.time() > deadline:
                break
            for trial in range(10):
                if time.time() > deadline:
                    break
                result = _als_decompose_full(target_rank, max_iters=50, deadline=deadline)
                if result:
                    if _verify(result[0], result[1], result[2], tol=1e-6):
                        best_U, best_V, best_W = result
                        best_rank = target_rank
                        break
            if best_rank == target_rank:
                break

    # Round the best result
    rounded = _round_and_verify(best_U, best_V, best_W)
    if rounded:
        best_U, best_V, best_W = rounded

    return {"U": best_U, "V": best_V, "W": best_W}
