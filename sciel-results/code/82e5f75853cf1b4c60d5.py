import time
import numpy as np

DIMENSIONS = [3, 5, 6]
n, m, p = DIMENSIONS
nm, mp, pn = n*m, m*p, p*n

def _target_tensor():
    T = np.zeros((nm, mp, pn), dtype=float)
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
    # Efficient reconstruction
    # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Using einsum
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.abs(recon - T)
    return np.max(diff) < tol

def _round_half_int(M, tol=1e-6):
    M2 = 2.0 * np.asarray(M, dtype=float)
    Mr = np.round(M2)
    if np.max(np.abs(M2 - Mr)) < tol:
        return Mr / 2.0, True
    return None, False

def _als_decompose(T_sub, R, max_iter=50, tol=1e-6, seed=0, time_limit=5.0):
    """
    ALS for a sub-tensor T_sub with dimensions (d1, d2, d3).
    Returns U, V, W if success, else None.
    """
    d1, d2, d3 = T_sub.shape
    rng = np.random.RandomState(seed)
    U = rng.randn(d1, R) * 0.5
    V = rng.randn(d2, R) * 0.5
    W = rng.randn(d3, R) * 0.5
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Update U
        # For each a, solve: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # Fix V, W.
        # M[b,c,r] = V[b,r] W[c,r]
        MW = np.einsum('br,cr->bcr', V, W, optimize=True)
        MW_flat = MW.reshape(d2*d3, R)
        MW_T = MW_flat.T
        for a in range(d1):
            target_a = T_sub[a].reshape(d2*d3)
            try:
                X, _, _, _ = np.linalg.lstsq(MW_T, target_a, rcond=None)
                U[a, :] = X
            except:
                pass
        
        # Update V
        UW = np.einsum('ar,cr->acr', U, W, optimize=True)
        UW_flat = UW.reshape(d1*d3, R)
        UW_T = UW_flat.T
        for b in range(d2):
            target_b = T_sub[:, b, :].reshape(d1*d3)
            try:
                X, _, _, _ = np.linalg.lstsq(UW_T, target_b, rcond=None)
                V[b, :] = X
            except:
                pass
        
        # Update W
        UV = np.einsum('ar,br->abr', U, V, optimize=True)
        UV_flat = UV.reshape(d1*d2, R)
        UV_T = UV_flat.T
        for c in range(d3):
            target_c = T_sub[:, :, c].reshape(d1*d2)
            try:
                X, _, _, _ = np.linalg.lstsq(UV_T, target_c, rcond=None)
                W[c, :] = X
            except:
                pass
        
        # Check error
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(recon - T_sub))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    
    if best_err < 1e-5:
        # Try rounding
        U_r, ok1 = _round_half_int(best_U, tol=1e-4)
        if ok1:
            V_r, ok2 = _round_half_int(best_V, tol=1e-4)
            if ok2:
                W_r, ok3 = _round_half_int(best_W, tol=1e-4)
                if ok3:
                    # Verify sub-tensor
                    recon_sub = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
                    if np.max(np.abs(recon_sub - T_sub)) < 1e-5:
                        return U_r, V_r, W_r
    return None

def _construct_block_decomp():
    """
    Decompose 3x5x6 into 3x2x6 and 3x3x6 blocks.
    Block 1: j in {0, 1}
    Block 2: j in {2, 3, 4}
    """
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # --- Block 1: 3 x 2 x 6 ---
    # Dimensions: n=3, m1=2, p=6
    # Indices:
    # a1 = i*2 + j1  (i in 0..2, j1 in 0..1) -> 6 entries
    # b1 = j1*6 + k  (j1 in 0..1, k in 0..5) -> 12 entries
    # c1 = k*3 + i   (k in 0..5, i in 0..2) -> 18 entries
    
    # Sub-tensor T1
    T1 = np.zeros((6, 12, 18), dtype=float)
    for i in range(3):
        for j1 in range(2):
            for k in range(6):
                a1 = i*2 + j1
                b1 = j1*6 + k
                c1 = k*3 + i
                T1[a1, b1, c1] = 1.0
    
    # --- Block 2: 3 x 3 x 6 ---
    # Dimensions: n=3, m2=3, p=6
    # Indices:
    # a2 = i*3 + j2  (i in 0..2, j2 in 0..2) -> 9 entries
    # b2 = (2+j2)*6 + k (j2 in 0..2, k in 0..5) -> 18 entries
    # c2 = k*3 + i   (k in 0..5, i in 0..2) -> 18 entries
    
    # Sub-tensor T2
    T2 = np.zeros((9, 18, 18), dtype=float)
    for i in range(3):
        for j2 in range(3):
            for k in range(6):
                j_actual = 2 + j2
                a2 = i*3 + j2
                b2 = j_actual*6 + k
                c2 = k*3 + i
                T2[a2, b2, c2] = 1.0

    # Solve Block 1
    U1, V1, W1 = None, None, None
    # Lower bound for 3x2x6 is max(6,12,18)=18. Try 18, 19, 20...
    for R1 in range(18, 30):
        if time.time() > deadline - 30: break
        for seed in range(5):
            if time.time() > deadline - 30: break
            res = _als_decompose(T1, R1, max_iter=30, seed=seed*10+R1, time_limit=3.0)
            if res:
                U1, V1, W1 = res
                break
        if U1 is not None:
            break
            
    if U1 is None:
        # Fallback to schoolbook for block 1: rank = 3*2*6 = 36
        R1 = 36
        U1 = np.zeros((6, R1))
        V1 = np.zeros((12, R1))
        W1 = np.zeros((18, R1))
        col = 0
        for i in range(3):
            for j1 in range(2):
                for k in range(6):
                    a1 = i*2 + j1
                    b1 = j1*6 + k
                    c1 = k*3 + i
                    U1[a1, col] = 1
                    V1[b1, col] = 1
                    W1[c1, col] = 1
                    col += 1

    # Solve Block 2
    U2, V2, W2 = None, None, None
    # Lower bound for 3x3x6 is max(9,18,18)=18. 
    # But 3x3x3 is rank 23. Two blocks -> 46.
    # Try ranks 36 (schoolbook 3*3*3*2) down to 40? 
    # Actually, let's try a range.
    for R2 in range(36, 55):
        if time.time() > deadline - 10: break
        for seed in range(3):
            if time.time() > deadline - 10: break
            res = _als_decompose(T2, R2, max_iter=30, seed=seed*100+R2, time_limit=3.0)
            if res:
                U2, V2, W2 = res
                break
        if U2 is not None:
            break
            
    if U2 is None:
        # Fallback to schoolbook for block 2: rank = 3*3*6 = 54
        R2 = 54
        U2 = np.zeros((9, R2))
        V2 = np.zeros((18, R2))
        W2 = np.zeros((18, R2))
        col = 0
        for i in range(3):
            for j2 in range(3):
                for k in range(6):
                    j_actual = 2 + j2
                    a2 = i*3 + j2
                    b2 = j_actual*6 + k
                    c2 = k*3 + i
                    U2[a2, col] = 1
                    V2[b2, col] = 1
                    W2[c2, col] = 1
                    col += 1

    # Combine into full U, V, W
    # Full U (15 x R_total)
    # Rows 0..5 correspond to Block 1 U1
    # Rows 6..14 correspond to Block 2 U2
    # Note: Mapping of indices
    # Block 1: a1 = i*2+j1. Full index a_full = i*5+j1.
    #   i=0, j1=0 -> a1=0, a_full=0
    #   i=0, j1=1 -> a1=1, a_full=1
    #   i=1, j1=0 -> a1=2, a_full=5
    #   i=1, j1=1 -> a1=3, a_full=6
    #   i=2, j1=0 -> a1=4, a_full=10
    #   i=2, j1=1 -> a1=5, a_full=11
    # Block 2: a2 = i*3+j2. Full index a_full = i*5 + (2+j2).
    #   i=0, j2=0 -> a2=0, a_full=2
    #   i=0, j2=1 -> a2=1, a_full=3
    #   i=0, j2=2 -> a2=2, a_full=4
    #   i=1, j2=0 -> a2=3, a_full=7
    #   i=1, j2=1 -> a2=4, a_full=8
    #   i=1, j2=2 -> a2=5, a_full=9
    #   i=2, j2=0 -> a2=6, a_full=12
    #   i=2, j2=1 -> a2=7, a_full=13
    #   i=2, j2=2 -> a2=8, a_full=14
    
    R_total = U1.shape[1] + U2.shape[1]
    U_full = np.zeros((15, R_total))
    V_full = np.zeros((30, R_total))
    W_full = np.zeros((18, R_total))
    
    # Place Block 1
    U_full[0:6, 0:U1.shape[1]] = U1
    V_full[0:12, 0:V1.shape[1]] = V1
    W_full[0:18, 0:W1.shape[1]] = W1
    
    # Place Block 2
    offset = U1.shape[1]
    # Map U2 rows to full U rows
    # U2 row i corresponds to full row i*5 + 2 + j2? No, U2 is indexed by a2=i*3+j2.
    # We need to scatter U2 into U_full.
    for i in range(3):
        for j2 in range(3):
            a2 = i*3 + j2
            a_full = i*5 + (2 + j2)
            U_full[a_full, offset:offset+U2.shape[1]] = U2[a2, :]
            
    # Map V2 rows to full V rows
    # V2 is indexed by b2 = j_actual*6 + k, where j_actual = 2+j2.
    # Full V index is b_full = j_actual*6 + k.
    # So V2 rows are already in the correct order for the bottom part of V_full.
    # V_full rows 18..29 correspond to j in 2,3,4.
    # V2 has 18 rows.
    V_full[18:36, offset:offset+V2.shape[1]] = V2
    
    # Map W2 rows to full W rows
    # W2 is indexed by c2 = k*3 + i.
    # Full W index is c_full = k*3 + i.
    # So W2 rows are identical to the full W rows.
    W_full[0:18, offset:offset+W2.shape[1]] = W2
    
    # Verify
    if _verify(U_full, V_full, W_full):
        return U_full, V_full, W_full
    
    # If verification fails (should not happen if sub-tensors are correct and disjoint), fallback
    return None

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

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Try block decomposition
    try:
        res = _construct_block_decomp()
        if res is not None:
            U, V, W = res
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except Exception as e:
        pass
    
    # Fallback to schoolbook
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
