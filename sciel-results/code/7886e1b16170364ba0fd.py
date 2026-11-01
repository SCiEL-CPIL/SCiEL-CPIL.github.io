"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# Precompute index structures for fast ALS
def _precompute_indices():
    n, m, p = DIMENSIONS
    # For each a = i*m+j, which (b,c) pairs have target value 1?
    # Target: T[a,b,c]=1 where a=i*m+j, b=j*p+k, c=k*n+i
    # For U update: fix a, vary k -> b=j*p+k, c=k*n+i
    U_indices = []  # U_indices[a] = list of (b, c) for k in range(p)
    for i in range(n):
        for j in range(m):
            a = i * m + j
            pairs = [(j * p + k, k * n + i) for k in range(p)]
            U_indices.append(pairs)
    
    # For V update: fix b = j*p+k, vary (i, kk) -> a=i*m+j, c=kk*n+i
    V_indices = []  # V_indices[b] = list of (a, c) for (i, kk)
    for j in range(m):
        for k in range(p):
            b = j * p + k
            pairs = [(i * m + j, kk * n + i) for i in range(n) for kk in range(p)]
            V_indices.append(pairs)
    
    # For W update: fix c = k*n+i, vary (ii, j) -> a=ii*m+j, b=j*p+k
    W_indices = []  # W_indices[c] = list of (a, b) for (ii, j)
    for k in range(p):
        for i in range(n):
            c = k * n + i
            pairs = [(ii * m + j, j * p + k) for ii in range(n) for j in range(m)]
            W_indices.append(pairs)
    
    # Target mask
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    return U_indices, V_indices, W_indices, target

_U_IDX, _V_IDX, _W_IDX, _TARGET = _precompute_indices()

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
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
    
    non_target_err = np.max(np.abs(T[~_TARGET])) if (~_TARGET).any() else 0.0
    
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

def _compute_residual(U, V, W):
    """Compute max error on target and non-target entries."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for r in range(U.shape[1]):
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
    
    non_target_err = np.max(np.abs(T[~_TARGET])) if (~_TARGET).any() else 0.0
    return max_err, non_target_err

def _als_decompose(rank, max_iters=50, deadline=None, seed=None):
    """
    Alternating Least Squares for tensor decomposition with precomputed indices.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    U = rng.randn(n*m, rank) * 0.5
    V = rng.randn(m*p, rank) * 0.5
    W = rng.randn(p*n, rank) * 0.5
    
    prev_err = float('inf')
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for a in range(n*m):
            pairs = _U_IDX[a]
            M = np.zeros((len(pairs), rank))
            for idx, (b, c) in enumerate(pairs):
                M[idx, :] = V[b, :] * W[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(pairs)), rcond=None)
                U[a, :] = x
            except:
                pass
        
        # Update V
        for b in range(m*p):
            pairs = _V_IDX[b]
            M = np.zeros((len(pairs), rank))
            for idx, (a, c) in enumerate(pairs):
                M[idx, :] = U[a, :] * W[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(pairs)), rcond=None)
                V[b, :] = x
            except:
                pass
        
        # Update W
        for c in range(p*n):
            pairs = _W_IDX[c]
            M = np.zeros((len(pairs), rank))
            for idx, (a, b) in enumerate(pairs):
                M[idx, :] = U[a, :] * V[b, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(pairs)), rcond=None)
                W[c, :] = x
            except:
                pass
        
        # Check convergence
        max_err, non_target_err = _compute_residual(U, V, W)
        cur_err = max(max_err, non_target_err)
        
        if cur_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
        
        if cur_err > prev_err * 1.5:
            # Diverging, stop
            break
        prev_err = cur_err
        
    return None

def _polish(U, V, W, max_iters=20, tol=1e-10, deadline=None):
    """
    Local refinement: do a few more ALS iterations with tighter tolerance.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for a in range(n*m):
            pairs = _U_IDX[a]
            M = np.zeros((len(pairs), rank))
            for idx, (b, c) in enumerate(pairs):
                M[idx, :] = V[b, :] * W[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(pairs)), rcond=None)
                U[a, :] = x
            except:
                pass
        
        # Update V
        for b in range(m*p):
            pairs = _V_IDX[b]
            M = np.zeros((len(pairs), rank))
            for idx, (a, c) in enumerate(pairs):
                M[idx, :] = U[a, :] * W[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(pairs)), rcond=None)
                V[b, :] = x
            except:
                pass
        
        # Update W
        for c in range(p*n):
            pairs = _W_IDX[c]
            M = np.zeros((len(pairs), rank))
            for idx, (a, b) in enumerate(pairs):
                M[idx, :] = U[a, :] * V[b, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(len(pairs)), rcond=None)
                W[c, :] = x
            except:
                pass
        
        max_err, non_target_err = _compute_residual(U, V, W)
        if max_err < tol and non_target_err < tol:
            break
    
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
    deadline = start_time + 135  # Safety margin
    
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
    
    # Now try to find better with ALS
    if best_rank > 40 and time.time() < deadline:
        # Try ranks from 32 to best_rank-1
        for rank in range(32, best_rank):
            if time.time() > deadline:
                break
            
            found = False
            # Try multiple seeds
            for seed in range(5):
                if time.time() > deadline:
                    break
                
                result = _als_decompose(rank, max_iters=40, deadline=deadline, seed=seed)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        # Polish for higher precision
                        U_arr = np.array(U)
                        V_arr = np.array(V)
                        W_arr = np.array(W)
                        polished = _polish(U_arr, V_arr, W_arr, max_iters=10, tol=1e-12, deadline=deadline)
                        if _verify(polished, polished, polished, tol=1e-6) or _verify(U, V, W, tol=1e-6):
                            rounded = _round_and_verify(U, V, W)
                            if rounded:
                                best_U, best_V, best_W = rounded
                                best_rank = rank
                            else:
                                best_U, best_V, best_W = U, V, W
                                best_rank = rank
                            found = True
                            break
            
            if found:
                # Try to go lower
                continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
