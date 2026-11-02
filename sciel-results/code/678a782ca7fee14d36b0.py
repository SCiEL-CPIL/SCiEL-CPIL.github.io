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
    Decompose 3x5x6 using block structure.
    Split: middle dim 5 -> 2+3, last dim 6 -> 3+3
    Blocks:
      A: 3x2x3 (i in 0..2, j in 0..1, k in 0..2)
      B: 3x2x3 (i in 0..2, j in 0..1, k in 3..5)
      C: 3x3x3 (i in 0..2, j in 2..4, k in 0..2)
      D: 3x3x3 (i in 0..2, j in 2..4, k in 3..5)
    Schoolbook ranks: 18, 18, 27, 27 = 90
    We can do better on 3x3x3 (rank 23) and 3x2x3 (rank <= 18)
    """
    start_time = time.time()
    deadline = 140.0
    
    # Try to find good decompositions for 3x3x3 and 3x2x3
    # For 3x3x3, schoolbook is 27, known optimal is 23
    # For 3x2x3, schoolbook is 18, lower bound is 18 (max(6,6,9)=9, but structural constraints)
    
    # Let's try to solve 3x3x3 at rank 23, 24, 25...
    # And 3x2x3 at rank 18, 19...
    
    # Build 3x3x3 sub-tensor
    T333 = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i*3 + j
                b = j*3 + k
                c = k*3 + i
                T333[a, b, c] = 1.0
    
    # Build 3x2x3 sub-tensor
    T323 = np.zeros((6, 6, 9), dtype=float)
    for i in range(3):
        for j in range(2):
            for k in range(3):
                a = i*2 + j
                b = j*3 + k
                c = k*3 + i
                T323[a, b, c] = 1.0
    
    # Solve 3x3x3
    U333, V333, W333 = None, None, None
    for R in range(23, 28):
        if time.time() > deadline - 40: break
        for seed in range(8):
            if time.time() > deadline - 40: break
            res = _als_decompose(T333, R, max_iter=40, seed=seed*100+R, time_limit=4.0)
            if res:
                U333, V333, W333 = res
                break
        if U333 is not None:
            break
    
    if U333 is None:
        # Fallback: schoolbook for 3x3x3, rank 27
        R = 27
        U333 = np.zeros((9, R))
        V333 = np.zeros((9, R))
        W333 = np.zeros((9, R))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i*3 + j
                    b = j*3 + k
                    c = k*3 + i
                    U333[a, col] = 1
                    V333[b, col] = 1
                    W333[c, col] = 1
                    col += 1
    
    # Solve 3x2x3
    U323, V323, W323 = None, None, None
    for R in range(18, 22):
        if time.time() > deadline - 20: break
        for seed in range(8):
            if time.time() > deadline - 20: break
            res = _als_decompose(T323, R, max_iter=40, seed=seed*100+R+1000, time_limit=4.0)
            if res:
                U323, V323, W323 = res
                break
        if U323 is not None:
            break
    
    if U323 is None:
        # Fallback: schoolbook for 3x2x3, rank 18
        R = 18
        U323 = np.zeros((6, R))
        V323 = np.zeros((6, R))
        W323 = np.zeros((9, R))
        col = 0
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    a = i*2 + j
                    b = j*3 + k
                    c = k*3 + i
                    U323[a, col] = 1
                    V323[b, col] = 1
                    W323[c, col] = 1
                    col += 1
    
    # Now assemble the full 3x5x6 decomposition
    # We have 4 blocks:
    # Block A: i in 0..2, j in 0..1, k in 0..2 -> uses U323, V323, W323 (with k offset 0)
    # Block B: i in 0..2, j in 0..1, k in 3..5 -> uses U323, V323, W323 (with k offset 3)
    # Block C: i in 0..2, j in 2..4, k in 0..2 -> uses U333, V333, W333 (with k offset 0)
    # Block D: i in 0..2, j in 2..4, k in 3..5 -> uses U333, V333, W333 (with k offset 3)
    
    R_A = U323.shape[1]
    R_B = U323.shape[1]
    R_C = U333.shape[1]
    R_D = U333.shape[1]
    R_total = R_A + R_B + R_C + R_D
    
    U_full = np.zeros((15, R_total))
    V_full = np.zeros((30, R_total))
    W_full = np.zeros((18, R_total))
    
    # Block A: j in 0..1, k in 0..2
    # U rows: i*5 + j for i in 0..2, j in 0..1 -> rows 0,1,5,6,10,11
    # V rows: j*6 + k for j in 0..1, k in 0..2 -> rows 0,1,2,6,7,8
    # W rows: k*3 + i for k in 0..2, i in 0..2 -> rows 0,1,2,3,4,5
    off = 0
    for i in range(3):
        for j in range(2):
            a_full = i*5 + j
            a_sub = i*2 + j
            U_full[a_full, off:off+R_A] = U323[a_sub, :]
    for j in range(2):
        for k in range(3):
            b_full = j*6 + k
            b_sub = j*3 + k
            V_full[b_full, off:off+R_A] = V323[b_sub, :]
    for k in range(3):
        for i in range(3):
            c_full = k*3 + i
            c_sub = k*3 + i
            W_full[c_full, off:off+R_A] = W323[c_sub, :]
    
    # Block B: j in 0..1, k in 3..5
    off += R_A
    for i in range(3):
        for j in range(2):
            a_full = i*5 + j
            a_sub = i*2 + j
            U_full[a_full, off:off+R_B] = U323[a_sub, :]
    for j in range(2):
        for k in range(3):
            k_sub = k  # 0,1,2
            b_full = j*6 + (3 + k_sub)
            b_sub = j*3 + k_sub
            V_full[b_full, off:off+R_B] = V323[b_sub, :]
    for k_sub in range(3):
        for i in range(3):
            c_full = (3 + k_sub)*3 + i
            c_sub = k_sub*3 + i
            W_full[c_full, off:off+R_B] = W323[c_sub, :]
    
    # Block C: j in 2..4, k in 0..2
    off += R_B
    for i in range(3):
        for j_sub in range(3):
            j_full = 2 + j_sub
            a_full = i*5 + j_full
            a_sub = i*3 + j_sub
            U_full[a_full, off:off+R_C] = U333[a_sub, :]
    for j_sub in range(3):
        for k in range(3):
            j_full = 2 + j_sub
            b_full = j_full*6 + k
            b_sub = j_sub*3 + k
            V_full[b_full, off:off+R_C] = V333[b_sub, :]
    for k in range(3):
        for i in range(3):
            c_full = k*3 + i
            c_sub = k*3 + i
            W_full[c_full, off:off+R_C] = W333[c_sub, :]
    
    # Block D: j in 2..4, k in 3..5
    off += R_C
    for i in range(3):
        for j_sub in range(3):
            j_full = 2 + j_sub
            a_full = i*5 + j_full
            a_sub = i*3 + j_sub
            U_full[a_full, off:off+R_D] = U333[a_sub, :]
    for j_sub in range(3):
        for k_sub in range(3):
            j_full = 2 + j_sub
            b_full = j_full*6 + (3 + k_sub)
            b_sub = j_sub*3 + k_sub
            V_full[b_full, off:off+R_D] = V333[b_sub, :]
    for k_sub in range(3):
        for i in range(3):
            c_full = (3 + k_sub)*3 + i
            c_sub = k_sub*3 + i
            W_full[c_full, off:off+R_D] = W333[c_sub, :]
    
    # Verify
    if _verify(U_full, V_full, W_full):
        return U_full, V_full, W_full
    
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
            rank = U.shape[1]
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
