"""Matrix multiplication 2x4 x 4x7: Strassen blocks + alternating minimization refinement."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
_DEADLINE = time.time() + 140  # safety margin under 150s

def _time_left():
    return _DEADLINE - time.time()

def _get_strassen_222():
    """Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    U: (4, 7) indexed by (i,j) -> i*2+j
    V: (4, 7) indexed by (j,k) -> j*2+k
    W: (4, 7) indexed by (k,i) -> k*2+i
    """
    r = 7
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W

def _get_school_221():
    """Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4)."""
    r = 4
    U = np.zeros((4, r))
    V = np.zeros((2, r))
    W = np.zeros((2, r))
    for c in range(r):
        i = c // 2
        j = c % 2
        U[i * 2 + j, c] = 1.0
        V[j, c] = 1.0
        W[i, c] = 1.0
    return U, V, W

def _build_base():
    """Build base Strassen block decomposition for 2x4x7, rank 50."""
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    for j_off in [0, 2]:
        for k_start, k_end, p_type in [(0, 2, 's'), (2, 4, 's'), (4, 6, 's'), (6, 7, 'c')]:
            if p_type == 's':
                U_sub, V_sub, W_sub, r_sub = U_s, V_s, W_s, 7
                p_sub = 2
            else:
                U_sub, V_sub, W_sub, r_sub = U_sc, V_sc, W_sc, 4
                p_sub = 1
            for i in range(n):
                for j_sub in range(2):
                    j = j_off + j_sub
                    U[i * m + j, col:col + r_sub] = U_sub[i * 2 + j_sub, :]
            for j_sub in range(2):
                j = j_off + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col:col + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col:col + r_sub] = W_sub[k_sub * 2 + i, :]
            col += r_sub
    return U, V, W

def _build_target():
    """Build target tensor T of shape (8, 28, 14)."""
    n, m, p = 2, 4, 7
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _tensor_from(U, V, W):
    """Compute tensor T[i,j,c] = sum_d U[i,d]*V[j,d]*W[c,d]."""
    return np.einsum('ad,bd,cd->abc', U, V, W)

def _verify(U, V, W, T, tol=1e-6):
    """Check if decomposition matches target tensor."""
    R = _tensor_from(U, V, W)
    return np.allclose(R, T, atol=tol, rtol=tol)

def _round_half(M):
    """Round to nearest half-integer."""
    return np.round(M * 2.0) / 2.0

def _alt_minimize(U, V, W, T, max_iter=200, tol=1e-8):
    """Alternating minimization: fix two, solve for third via least squares."""
    a, b, c = T.shape
    r = U.shape[1]
    T_flat = T.reshape(a * b * c)
    
    for it in range(max_iter):
        if _time_left() < 5:
            return None
        
        # Optimize W: T[i,j,c] = sum_d U[i,d]*V[j,d]*W[c,d]
        # For each c, d: coeff = U[:,d] * V[:,d] (outer product flattened)
        # D_w has shape (a*b*c, r): D_w[(i*b+j)*c+c_val, d] = U[i,d]*V[j,d]
        # More efficiently: for each d, outer(U[:,d], V[:,d]) tiled c times
        D_w = np.zeros((a * b * c, r))
        for d in range(r):
            outer_d = np.outer(U[:, d], V[:, d]).reshape(a * b)
            D_w[:, d] = np.tile(outer_d, c)
        W_new = np.linalg.lstsq(D_w, T_flat, rcond=None)[0].reshape(c, r)
        
        # Optimize U: T[i,j,c] = sum_d U[i,d]*V[j,d]*W[c,d]
        # D_u[(i*b+j)*c+c_val, d] = V[j,d]*W[c_val,d]
        D_u = np.zeros((a * b * c, r))
        for d in range(r):
            # For each (j, c_val): V[j,d]*W[c_val,d], repeated a times
            vw = np.outer(W[:, d], V[:, d]).reshape(c * b)  # c_val, j
            D_u[:, d] = np.tile(vw.T, (a, 1)).reshape(a * b * c)
        U_new = np.linalg.lstsq(D_u, T_flat, rcond=None)[0].reshape(a, r)
        
        # Optimize V: T[i,j,c] = sum_d U[i,d]*V[j,d]*W[c,d]
        # D_v[(i*b+j)*c+c_val, d] = U[i,d]*W[c_val,d]
        D_v = np.zeros((a * b * c, r))
        for d in range(r):
            uw = np.outer(U_new[:, d], W_new[:, d]).reshape(a * c)  # i, c_val
            D_v[:, d] = np.tile(uw.T, (b, 1)).reshape(a * b * c)
        V_new = np.linalg.lstsq(D_v, T_flat, rcond=None)[0].reshape(b, r)
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        R = _tensor_from(U, V, W)
        err = np.max(np.abs(R - T))
        if err < tol:
            break
    
    if _verify(U, V, W, T, tol=1e-4):
        return U, V, W
    return None

def _try_compression(U, V, W, T, target_rank, n_trials=5, seed=42):
    """Try to compress to target_rank by random projection + alt min."""
    a, b, c = T.shape
    r_orig = U.shape[1]
    rng = np.random.RandomState(seed)
    
    for trial in range(n_trials):
        if _time_left() < 10:
            break
        
        # Random projection to target_rank
        P = rng.randn(r_orig, target_rank)
        U_p = U @ P
        V_p = V @ P
        W_p = W @ P
        
        result = _alt_minimize(U_p, V_p, W_p, T, max_iter=300)
        if result is not None:
            U_r, V_r, W_r = result
            # Round to half-integers and verify
            U_h = _round_half(U_r)
            V_h = _round_half(V_r)
            W_h = _round_half(W_r)
            if _verify(U_h, V_h, W_h, T, tol=1e-6):
                return U_h, V_h, W_h
            # Also try without rounding
            if _verify(U_r, V_r, W_r, T, tol=1e-6):
                return U_r, V_r, W_r
    return None

def construct():
    n, m, p = DIMENSIONS
    U0, V0, W0 = _build_base()
    T = _build_target()
    
    # Verify base solution
    if not _verify(U0, V0, W0, T, tol=1e-6):
        # Fallback to schoolbook
        r = n * m * p
        U_fb = np.zeros((n * m, r))
        V_fb = np.zeros((m * p, r))
        W_fb = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = 50
    
    # Try to compress to lower ranks
    # Theoretical lower bound: for 2x4x7, rank >= max(2*4, 4*7, 7*2) = 28
    # But Strassen-based gives 50. Try ranks from 40 down to 28.
    for target_rank in range(40, 27, -1):
        if _time_left() < 15:
            break
        result = _try_compression(U0, V0, W0, T, target_rank, n_trials=3, seed=target_rank)
        if result is not None:
            U_r, V_r, W_r = result
            if V_r.shape[1] < best_rank:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = V_r.shape[1]
                break  # Got a lower rank, stop
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
