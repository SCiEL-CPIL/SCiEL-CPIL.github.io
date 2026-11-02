import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

_T_cache = None
def _get_T():
    global _T_cache
    if _T_cache is None:
        _T_cache = _target_tensor()
    return _T_cache

def _verify(U, V, W, tol=1e-8):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = _get_T()
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.abs(recon - T)
    return np.max(diff) < tol

def _round_half_int(M, tol=1e-6):
    M2 = 2.0 * np.asarray(M, dtype=float)
    Mr = np.round(M2)
    if np.max(np.abs(M2 - Mr)) < tol:
        return Mr / 2.0, True
    return None, False

def _construct_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's rank-7 method.
    U: 4x7, V: 4x7, W: 4x7
    T[i*2+j, j*2+k, k*2+i] = 1
    """
    # Strassen's 7 products for 2x2x2
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    
    # The tensor T[a,b,c] where a = i*2+j, b = j*2+k, c = k*2+i
    # T is 1 when (a,b,c) corresponds to (i,j,k) for valid i,j,k in {0,1}
    
    # Strassen decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B00-B10)
    # M4 = A11(B11-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, we need:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Let me construct this explicitly.
    # a = i*2+j, b = j*2+k, c = k*2+i
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # A00: a=0 (i=0,j=0), A11: a=3 (i=1,j=1)
    # B00: b=0 (j=0,k=0), B11: b=3 (j=1,k=1)
    # C00: c=0 (k=0,i=0), C11: c=3 (k=1,i=1)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A10+A11)B00
    # A10: a=2 (i=1,j=0), A11: a=3 (i=1,j=1)
    # B00: b=0 (j=0,k=0)
    # C10: c=2 (k=0,i=1)
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # M3 = A00(B00-B10)
    # A00: a=0, B00: b=0, B10: b=2 (j=1,k=0)
    # C01: c=1 (k=1,i=0)
    U[0,2] = 1
    V[0,2] = 1; V[2,2] = -1
    W[1,2] = 1
    
    # M4 = A11(B11-B01)
    # A11: a=3, B11: b=3, B01: b=1 (j=0,k=1)
    # C00: c=0, C11: c=3
    U[3,3] = 1
    V[3,3] = 1; V[1,3] = -1
    W[0,3] = 1; W[3,3] = 1
    
    # M5 = (A00+A01)B11
    # A00: a=0, A01: a=1 (i=0,j=1), B11: b=3
    # C01: c=1, C11: c=3
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1; W[3,4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    # A10: a=2, A00: a=0, B00: b=0, B01: b=1
    # C11: c=3
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # A01: a=1, A11: a=3, B10: b=2, B11: b=3
    # C00: c=0
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _construct_block_decomp():
    """
    Construct a decomposition for 3x5x6 using block methods.
    Split middle dim 5 = 2 + 3.
    Block 1: 3x2x6 (left part of A, top part of B)
    Block 2: 3x3x6 (left part of A, bottom part of B)
    
    For 3x2x6: Use Strassen on the 2x2 core
    For 3x3x6: Use schoolbook or better
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Split: m = m1 + m2 = 2 + 3
    m1, m2 = 2, 3
    
    # Block 1: 3 x 2 x 6
    # A1 is 3x2, B1 is 2x6
    # We can decompose this using Strassen-like approach
    
    # Block 2: 3 x 3 x 6
    # A2 is 3x3, B2 is 3x6
    # Use schoolbook for simplicity: rank = 3*3*6 = 54
    
    # For block 1 (3x2x6):
    # Think of it as: for each i in 0..2, k in 0..5:
    #   C[i,k] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k]
    # This is a 2-term sum. We can use a rank-2 decomposition per (i,k) pair,
    # but that gives 3*6*2 = 36 terms.
    # Better: use Strassen on the 2x2 part.
    
    # Actually, for 3x2x6, the tensor has:
    # a = i*2+j for i in 0..2, j in 0..1 (6 entries)
    # b = j*6+k for j in 0..1, k in 0..5 (12 entries)  
    # Wait, that's not right. Let me re-derive.
    
    # For full 3x5x6:
    # a = i*5+j for i in 0..2, j in 0..4 (15 entries)
    # b = j*6+k for j in 0..4, k in 0..5 (30 entries)
    # c = k*3+i for k in 0..5, i in 0..2 (18 entries)
    
    # Split j into j1 in 0..1 (first 2) and j2 in 2..4 (last 3)
    # Let j = j1 for j1 in 0,1, and j = 2 + j2' for j2' in 0,1,2
    
    # Block 1 (j in 0,1): 3x2x6
    # a1 = i*2+j1, b1 = j1*6+k, c1 = k*3+i
    # This is like 3x2x6 matrix mult
    
    # Block 2 (j in 2,3,4): 3x3x6
    # a2 = i*3+j2', b2 = (2+j2')*6+k, c2 = k*3+i
    # This is like 3x3x6 matrix mult
    
    # For block 1 (3x2x6), I'll use a rank-12 construction:
    # For each i in 0..2, j1 in 0..1, k in 0..5:
    #   Term: U[i*2+j1, r] * V[j1*6+k, r] * W[k*3+i, r]
    # Schoolbook would be 3*2*6 = 36 terms.
    # Can we do better? The lower bound is max(6, 12, 18) = 18.
    
    # Let me just use schoolbook for both blocks for now:
    # Block 1: 36 terms
    # Block 2: 54 terms
    # Total: 90 terms (same as full schoolbook)
    
    # That's not helpful. Let me try to do better on block 1.
    
    # For 3x2x6, note that it's equivalent to multiplying (3x2) by (2x6).
    # We can write this as: C = A1 * B1 where A1 is 3x2, B1 is 2x6.
    # C[i,k] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k]
    
    # One approach: use the fact that 2x2 matrix mult has rank 7 (Strassen).
    # But here we have 3x2 and 2x6, not 2x2 and 2x2.
    
    # Alternative: decompose into 3 independent 2x6 multiplications? No.
    
    # Let me try: for each i, C[i,:] = A1[i,0]*B1[0,:] + A1[i,1]*B1[1,:]
    # This is a linear combination of 2 rows of B1.
    
    # I'll use a rank-18 construction for 3x2x6 (the lower bound):
    # Actually, let me just implement schoolbook for now and get a working solution,
    # then try to improve.
    
    rank = 90
    U = np.zeros((15, rank))
    V = np.zeros((30, rank))
    W = np.zeros((18, rank))
    
    col = 0
    for i in range(3):
        for j in range(5):
            for k in range(6):
                a = i*5+j
                b = j*6+k
                c = k*3+i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    return U, V, W

def _to_list(M):
    M = np.asarray(M, dtype=float)
    out = []
    for row in M:
        out_row = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                out_row.append(int(round(val)))
            elif abs(val * 2 - round(val * 2)) < 1e-9:
                out_row.append(round(val * 2) / 2.0)
            else:
                out_row.append(val)
        out.append(out_row)
    return out

def _als_decompose(T, R, n, m, p, max_iter=20, tol=1e-6, seed=None, time_limit=8.0):
    """
    Alternating least squares for tensor decomposition.
    """
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    
    rng = np.random.RandomState(seed)
    U = rng.randn(nm, R) * 0.5
    V = rng.randn(mp, R) * 0.5
    W = rng.randn(pn, R) * 0.5
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Update U
        MW = np.einsum('br,cr->bcr', V, W, optimize=True)
        MW_flat = MW.reshape(mp*pn, R)
        MW_T = MW_flat.T
        for a in range(nm):
            target_a = T[a].reshape(mp*pn)
            try:
                X, _, _, _ = np.linalg.lstsq(MW_T, target_a, rcond=None)
                U[a, :] = X
            except:
                pass
        
        # Update V
        UW = np.einsum('ar,cr->acr', U, W, optimize=True)
        UW_flat = UW.reshape(nm*pn, R)
        UW_T = UW_flat.T
        for b in range(mp):
            target_b = T[:, b, :].reshape(nm*pn)
            try:
                X, _, _, _ = np.linalg.lstsq(UW_T, target_b, rcond=None)
                V[b, :] = X
            except:
                pass
        
        # Update W
        UV = np.einsum('ar,br->abr', U, V, optimize=True)
        UV_flat = UV.reshape(nm*mp, R)
        UV_T = UV_flat.T
        for c in range(pn):
            target_c = T[:, :, c].reshape(nm*mp)
            try:
                X, _, _, _ = np.linalg.lstsq(UV_T, target_c, rcond=None)
                W[c, :] = X
            except:
                pass
        
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(recon - T))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    
    if best_err < 1e-5:
        U_r, ok1 = _round_half_int(best_U, tol=1e-4)
        if ok1:
            V_r, ok2 = _round_half_int(best_V, tol=1e-4)
            if ok2:
                W_r, ok3 = _round_half_int(best_W, tol=1e-4)
                if ok3:
                    if _verify(U_r, V_r, W_r):
                        return U_r, V_r, W_r
    return None

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = 140.0
    n, m, p = DIMENSIONS
    T = _get_T()
    
    # Try constructive block decomposition first
    try:
        U, V, W = _construct_block_decomp()
        if _verify(U, V, W):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except:
        pass
    
    # Try ALS for ranks 85 down to 70
    for R in range(85, 69, -1):
        if time.time() > deadline - 5:
            break
        
        n_restarts = 2
        for restart in range(n_restarts):
            if time.time() > deadline - 3:
                break
            time_left = deadline - time.time() - 2
            t_limit = min(8.0, time_left / max(1, n_restarts - restart))
            if t_limit < 2.0:
                break
            
            result = _als_decompose(T, R, n, m, p, max_iter=15, tol=1e-6, seed=restart*100+R, time_limit=t_limit)
            if result is not None:
                U, V, W = result
                return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    # Fallback to schoolbook
    return _construct_schoolbook()
