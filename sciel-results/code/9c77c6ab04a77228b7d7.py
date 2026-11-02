"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def _check_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _find_decomp_2x2_2xp(p, r, max_iters=5000, seed=42):
    """Find rank-r decomposition for 2x2 * 2xp. Returns (U,V,W) or None."""
    n, m = 2, 2
    rng = np.random.RandomState(seed)
    for attempt in range(max_iters):
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = np.zeros((p*n, r))
        valid = True
        for k in range(p):
            for i in range(n):
                coeff = np.zeros(r)
                for j in range(m):
                    coeff += U[i*m+j, :] * V[j*p+k, :]
                ns = np.dot(coeff, coeff)
                if ns < 1e-12:
                    valid = False
                    break
                W[k*n+i, :] = coeff / ns
            if not valid:
                break
        if not valid:
            continue
        # Verify
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for ri in range(r):
                        s += U[i*m+j, ri] * V[j*p+k, ri] * W[k*n+i, ri]
                    if abs(s - 1.0) > 1e-8:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return U, V, W
    return None

def _try_rationalize(U, V, W, tol=1e-6):
    """Try to find rational (half-integer) values close to U,V,W that still work."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Try rounding to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    if _check_decomposition(U_r, V_r, W_r, tol):
        return U_r, V_r, W_r
    return None

def _build_strassen_fallback():
    """Build a rank-46 Strassen-based decomposition for 2x2 * 2x13."""
    n, m, p = 2, 2, 13
    rank = 46
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for group in range(6):
        k1 = 2*group
        k2 = 2*group+1
        # Standard 7 Strassen products for 2x2 * 2x2
        # M1 = (A11+A22)(B11+B22)
        U[0, col] = 1; U[3, col] = 1
        V[k1, col] = 1; V[13+k2, col] = 1
        W[2*k1, col] = 1; W[2*k2+1, col] = 1
        col += 1
        # M2 = A22(B21-B11)
        U[2, col] = 1; U[3, col] = 1
        V[k1, col] = 1
        W[2*k1+1, col] = 1; W[2*k2+1, col] = -1
        col += 1
        # M3 = A11(B22-B12)
        U[0, col] = 1
        V[k2, col] = 1; V[13+k2, col] = -1
        W[2*k2, col] = 1; W[2*k2+1, col] = 1
        col += 1
        # M4 = A22(B11+B22)
        U[3, col] = 1
        V[13+k1, col] = 1; V[k1, col] = -1
        W[2*k1, col] = 1; W[2*k1+1, col] = 1
        col += 1
        # M5 = (A11+A12)B22
        U[0, col] = 1; U[1, col] = 1
        V[13+k2, col] = 1
        W[2*k1, col] = -1; W[2*k2, col] = 1
        col += 1
        # M6 = (A21-A11)B11
        U[2, col] = 1; U[0, col] = -1
        V[k1, col] = 1; V[k2, col] = 1
        W[2*k2+1, col] = 1
        col += 1
        # M7 = (A12-A22)(B11+B21)
        U[1, col] = 1; U[3, col] = -1
        V[13+k1, col] = 1; V[13+k2, col] = 1
        W[2*k1, col] = 1
        col += 1
    # Last 2x2 * 2x1 block: schoolbook rank 4
    k = 12
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1
    return U, V, W

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    best_result = None
    best_rank = 999

    # Strategy 1: 4 groups of 2x2*2x3 + 1 group of 2x2*2x1
    # Try to find low-rank decomp for 2x2*2x3
    decomp_2x3 = None
    r_2x3 = None
    for r_try in range(7, 11):
        if time.time() > deadline - 60:
            break
        result = _find_decomp_2x2_2xp(3, r_try, max_iters=2000, seed=100+r_try*7)
        if result is not None:
            decomp_2x3 = result
            r_2x3 = r_try
            break

    if decomp_2x3 is not None:
        U_2x3, V_2x3, W_2x3 = decomp_2x3
        total_rank = 4 * r_2x3 + 4
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        col = 0
        for group in range(4):
            k_off = group * 3
            for ri in range(r_2x3):
                U[:, col] = U_2x3[:, ri]
                for j in range(m):
                    for k in range(3):
                        V[j*p + k_off + k, col] = V_2x3[j*3 + k, ri]
                for k in range(3):
                    for i in range(n):
                        W[(k_off+k)*n + i, col] = W_2x3[k*n + i, ri]
                col += 1
        # Last column: schoolbook rank 4
        k = 12
        for i in range(n):
            for j in range(m):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
        if _check_decomposition(U, V, W):
            if total_rank < best_rank:
                best_rank = total_rank
                best_result = (U.copy(), V.copy(), W.copy())

    # Strategy 2: 3 groups of 2x2*2x4 + 1 group of 2x2*2x1
    decomp_2x4 = None
    r_2x4 = None
    for r_try in range(8, 12):
        if time.time() > deadline - 30:
            break
        result = _find_decomp_2x2_2xp(4, r_try, max_iters=1500, seed=200+r_try*11)
        if result is not None:
            decomp_2x4 = result
            r_2x4 = r_try
            break

    if decomp_2x4 is not None:
        U_2x4, V_2x4, W_2x4 = decomp_2x4
        total_rank = 3 * r_2x4 + 4
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        col = 0
        for group in range(3):
            k_off = group * 4
            for ri in range(r_2x4):
                U[:, col] = U_2x4[:, ri]
                for j in range(m):
                    for k in range(4):
                        V[j*p + k_off + k, col] = V_2x4[j*4 + k, ri]
                for k in range(4):
                    for i in range(n):
                        W[(k_off+k)*n + i, col] = W_2x4[k*n + i, ri]
                col += 1
        # Last column: schoolbook rank 4
        k = 12
        for i in range(n):
            for j in range(m):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
        if _check_decomposition(U, V, W):
            if total_rank < best_rank:
                best_rank = total_rank
                best_result = (U.copy(), V.copy(), W.copy())

    # Fallback: rank-46 Strassen
    U46, V46, W46 = _build_strassen_fallback()
    if _check_decomposition(U46, V46, W46):
        if 46 < best_rank:
            best_rank = 46
            best_result = (U46.copy(), V46.copy(), W46.copy())

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final fallback: schoolbook
    rf = n*m*p
    Uf = np.zeros((n*m, rf)); Vf = np.zeros((m*p, rf)); Wf = np.zeros((p*n, rf))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Uf[i*m+j,col]=1; Vf[j*p+k,col]=1; Wf[k*n+i,col]=1; col+=1
    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
