import time
import numpy as np

DIMENSIONS = [3, 5, 6]
n, m, p = DIMENSIONS
nm, mp, pn = n * m, m * p, p * n
RANK_SCHOOLBOOK = nm * mp * pn  # 90


def _target_tensor():
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


_T_cache = None


def _get_T():
    global _T_cache
    if _T_cache is None:
        _T_cache = _target_tensor()
    return _T_cache


def _verify(U, V, W, tol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = _get_T()
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.max(np.abs(recon - T)) < tol


def _round_half_int(M, tol=1e-6):
    M2 = 2.0 * np.asarray(M, dtype=float)
    Mr = np.round(M2)
    if np.max(np.abs(M2 - Mr)) < tol:
        return Mr / 2.0, True
    return None, False


def _als_decompose(T, R, max_iter=300, tol=1e-8, seed=0):
    """
    Vectorized ALS for tensor decomposition.
    U (nm x R), V (mp x R), W (pn x R)
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(nm, R) * 0.5
    V = rng.randn(mp, R) * 0.5
    W = rng.randn(pn, R) * 0.5

    for it in range(max_iter):
        # Update U: fix V, W
        # For each a: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # M[b,c,r] = V[b,r] * W[c,r], shape (mp, pn, R)
        # M_flat shape (mp*pn, R)
        # We want: for each a, U[a,:] such that M_flat.T @ U[a,:] = T[a,:,:].flatten()
        # Stacked: M_flat.T (R x mp*pn) @ U.T (R x nm) = T.reshape(nm, mp*pn).T (mp*pn x nm)
        MW = np.einsum('br,cr->bcr', V, W, optimize=True)  # (mp, pn, R)
        MW_flat = MW.reshape(mp * pn, R)
        # Solve: A @ X = B where A = MW_flat.T (R x mp*pn), X = U.T (R x nm), B = T.reshape(nm, mp*pn).T (mp*pn x nm)
        A_u = MW_flat.T  # (R, mp*pn)
        B_u = T.reshape(nm, mp * pn).T  # (mp*pn, nm)
        # Solve A_u @ X = B_u in least squares
        # A_u is R x K, X is R x nm, B_u is K x nm
        # This is underdetermined in X (R < K typically), so use lstsq
        U_new = np.linalg.lstsq(A_u, B_u, rcond=None)[0].T  # (nm, R)
        U = U_new

        # Update V: fix U, W
        # For each b: sum_r V[b,r] U[a,r] W[c,r] = T[a,b,c]
        # N[a,c,r] = U[a,r] * W[c,r], shape (nm, pn, R)
        # N_flat shape (nm*pn, R)
        # A_v = N_flat.T (R x nm*pn), B_v = T[:,b,:].reshape(nm*pn) for each b
        # Stacked: A_v @ V.T = T.reshape(mp, nm*pn).T
        UW = np.einsum('ar,cr->acr', U, W, optimize=True)  # (nm, pn, R)
        UW_flat = UW.reshape(nm * pn, R)
        A_v = UW_flat.T  # (R, nm*pn)
        B_v = T.reshape(mp, nm * pn).T  # (nm*pn, mp)
        V_new = np.linalg.lstsq(A_v, B_v, rcond=None)[0].T  # (mp, R)
        V = V_new

        # Update W: fix U, V
        # For each c: sum_r W[c,r] U[a,r] V[b,r] = T[a,b,c]
        # P[a,b,r] = U[a,r] * V[b,r], shape (nm, mp, R)
        # P_flat shape (nm*mp, R)
        # A_w = P_flat.T (R x nm*mp), B_w = T[:,:,c].reshape(nm*mp) for each c
        # Stacked: A_w @ W.T = T.reshape(pn, nm*mp).T
        UV = np.einsum('ar,br->abr', U, V, optimize=True)  # (nm, mp, R)
        UV_flat = UV.reshape(nm * mp, R)
        A_w = UV_flat.T  # (R, nm*mp)
        B_w = T.reshape(pn, nm * mp).T  # (nm*mp, pn)
        W_new = np.linalg.lstsq(A_w, B_w, rcond=None)[0].T  # (pn, R)
        W = W_new

        # Check convergence
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(recon - T))
        if err < tol:
            break

    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    err = np.max(np.abs(recon - T))
    if err < 1e-6:
        U_r, ok1 = _round_half_int(U, tol=1e-4)
        if ok1:
            V_r, ok2 = _round_half_int(V, tol=1e-4)
            if ok2:
                W_r, ok3 = _round_half_int(W, tol=1e-4)
                if ok3:
                    if _verify(U_r, V_r, W_r):
                        return U_r, V_r, W_r
    return None


def _construct_schoolbook():
    rank = RANK_SCHOOLBOOK
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _to_int_or_half(lst):
    out = []
    for row in lst:
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
    deadline = 140.0
    T = _get_T()

    # Try ranks from low to high
    # Lower bound: max(nm, mp, pn) = 30, but tensor rank is likely higher
    # Known: rank of 3x5x6 matrix mult is at most 90 (schoolbook)
    # Try from 40 up to 90
    for R in range(40, RANK_SCHOOLBOOK + 1):
        if time.time() > deadline - 20.0:
            break

        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break

            result = _als_decompose(T, R, max_iter=200, tol=1e-8, seed=restart * 1000 + R)
            if result is not None:
                U_r, V_r, W_r = result
                return {
                    "U": _to_int_or_half(U_r.tolist()),
                    "V": _to_int_or_half(V_r.tolist()),
                    "W": _to_int_or_half(W_r.tolist())
                }

    # Fallback to schoolbook
    return _construct_schoolbook()
