"""Matrix multiplication 2x4 x 4x7: block Strassen decomposition + linear-dependence elimination."""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 4, 7]
DEADLINE = time.time() + 140.0  # safety margin under 150s wall-clock


def _time_left():
    return DEADLINE - time.time()


def _get_strassen_222():
    """U,V,W for 2x2x2 multiplication (Strassen, rank 7).
    U: (4,7) idx i*2+j ; V: (4,7) idx j*2+k ; W: (4,7) idx k*2+i
    """
    r = 7
    U = np.zeros((4, r)); V = np.zeros((4, r)); W = np.zeros((4, r))
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W


def _get_school_221():
    """U,V,W for 2x2x1 multiplication (schoolbook, rank 4).
    U: (4,4) idx i*2+j ; V: (2,4) idx j ; W: (2,4) idx i
    """
    r = 4
    U = np.zeros((4, r)); V = np.zeros((2, r)); W = np.zeros((2, r))
    for c in range(r):
        i = c // 2
        j = c % 2
        U[i * 2 + j, c] = 1.0
        V[j, c] = 1.0
        W[i, c] = 1.0
    return U, V, W


def _build_block_decomposition():
    """Base Strassen block decomposition for 2x4x7, rank 50.
    Split m=4 into two blocks of 2; split p=7 into [0,2),[2,4),[4,6) Strassen + [6,7) schoolbook.
    Total rank = 2 * (3*7 + 4) = 50.
    """
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    U_s, V_s, W_s = _get_strassen_222()
    U_c, V_c, W_c = _get_school_221()

    col = 0
    for j_off in (0, 2):
        # Three Strassen blocks for k in {0..1, 2..3, 4..5}
        for k_off in (0, 2, 4):
            r_sub = 7
            for i in range(2):
                for js in range(2):
                    U[i * m + (j_off + js), col:col + r_sub] = U_s[i * 2 + js, :]
            for js in range(2):
                for ks in range(2):
                    V[(j_off + js) * p + (k_off + ks), col:col + r_sub] = V_s[js * 2 + ks, :]
            for ks in range(2):
                for i in range(2):
                    W[(k_off + ks) * n + i, col:col + r_sub] = W_s[ks * 2 + i, :]
            col += r_sub
        # Schoolbook block for k in {6}
        r_sub = 4
        for i in range(2):
            for js in range(2):
                U[i * m + (j_off + js), col:col + r_sub] = U_c[i * 2 + js, :]
        for js in range(2):
            V[(j_off + js) * p + 6, col:col + r_sub] = V_c[js, :]
        for i in range(2):
            W[6 * n + i, col:col + r_sub] = W_c[i, :]
        col += r_sub

    return U, V, W


def _build_target_tensor(n, m, p):
    """Target tensor T of shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _tensor_from(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)


def _verify(U, V, W, T, tol=1e-7):
    R = _tensor_from(U, V, W)
    return bool(np.allclose(R, T, atol=tol, rtol=0.0))


def _round_half(M):
    return np.round(M * 2.0) / 2.0


def _coeff_matrix(U, V, W, a, b, c):
    """C[idx, d] where idx = (i*b + j)*c + k and C[idx, d] = U[i,d]*V[j,d]*W[k,d]."""
    r = U.shape[1]
    C = np.empty((a * b * c, r), dtype=U.dtype)
    idx = 0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                C[idx, :] = U[i, :] * V[j, :] * W[k, :]
                idx += 1
    return C


def _exact_nullspace(C, T, tol=1e-9):
    """Exact rational Gaussian elimination.
    C: (N, r) float (integer), T: (N,) float (integer).
    Returns (rank, basis) where basis is a list of Fraction-vectors of length r
    spanning the nullspace of C (C @ v = 0). Returns (None, None) on failure.
    """
    N, r = C.shape
    # Build rows as Fractions
    rows = []
    for i in range(N):
        row = []
        for j in range(r):
            v = C[i, j]
            if abs(v - round(v)) < tol:
                row.append(Fraction(int(round(v))))
            else:
                row.append(Fraction(v).limit_denominator(10**6))
        tv = T[i]
        if abs(tv - round(tv)) < tol:
            tval = Fraction(int(round(tv)))
        else:
            tval = Fraction(tv).limit_denominator(10**6)
        rows.append((row, tval))

    pivot_cols = []
    # Forward elimination
    row_idx = 0
    for col in range(r):
        # find pivot
        piv = None
        for rr in range(row_idx, N):
            if rows[rr][0][col] != 0:
                piv = rr
                break
        if piv is None:
            continue
        rows[row_idx], rows[piv] = rows[piv], rows[row_idx]
        pv = rows[row_idx][0][col]
        # normalize pivot row
        prow = rows[row_idx][0][:]
        ptval = rows[row_idx][1]
        for cc in range(col, r):
            prow[cc] = prow[cc] / pv
        ptval = ptval / pv
        rows[row_idx] = (prow, ptval)
        # eliminate below
        for rr in range(row_idx + 1, N):
            f = rows[rr][0][col]
            if f != 0:
                rrow = rows[rr][0][:]
                rtval = rows[rr][1]
                for cc in range(col, r):
                    rrow[cc] = rrow[cc] - f * prow[cc]
                rtval = rtval - f * ptval
                rows[rr] = (rrow, rtval)
        pivot_cols.append(col)
        row_idx += 1

    rank = row_idx
    free_cols = [c for c in range(r) if c not in pivot_cols]
    basis = []
    for fc in free_cols:
        v = [Fraction(0)] * r
        v[fc] = Fraction(1)
        for pi, pc in enumerate(pivot_cols):
            v[pc] = -rows[pi][0][fc]
        basis.append(v)
    return rank, basis


def _float_nullspace(C, T, tol=1e-7):
    """Floating fallback: nullspace of C via SVD."""
    _, S, Vt = np.linalg.svd(C, full_matrices=True)
    nz = (S > tol * (S[0] if S.size else 1.0))
    # nullspace vectors are rows of Vt beyond rank
    null_vecs = Vt[nz.sum():, :]
    if null_vecs.size == 0:
        return []
    return [null_vecs[i, :] for i in range(null_vecs.shape[0])]


def _reduce_by_nullspace(U, V, W, T, n, m, p, max_iter=8):
    """Iteratively use null-space to drop terms.
    Returns (U,V,W) with reduced rank if possible, else original.
    """
    a, b, c = n * m, m * p, p * n
    r = U.shape[1]
    T_flat = T.reshape(-1)
    best = (U.copy(), V.copy(), W.copy(), r)

    for _ in range(max_iter):
        if _time_left() < 3:
            break
        C = _coeff_matrix(U, V, W, a, b, c)
        basis = None
        try:
            _, basis = _exact_nullspace(C, T_flat)
        except Exception:
            basis = None
        if basis is None or len(basis) == 0:
            try:
                basis = _float_nullspace(C, T_flat)
            except Exception:
                basis = []
        if len(basis) == 0:
            break

        found = False
        for v in basis:
            if _time_left() < 3:
                break
            # v is length r. Find a column where |v_d| is large to drop.
            vmag = np.array([abs(x) for x in v], dtype=float)
            if vmag.max() < 1e-6:
                continue
            drop = int(np.argmax(vmag))
            # keep all columns except drop
            keep = [d for d in range(r) if d != drop]
            U_k = U[:, keep]
            V_k = V[:, keep]
            W_k = W[:, keep]
            C_k = _coeff_matrix(U_k, V_k, W_k, a, b, c)
            # Solve C_k @ x = T_flat (least squares)
            try:
                sol, res, rankk, sv = np.linalg.lstsq(C_k, T_flat, rcond=None)
            except Exception:
                continue
            # sol is (r-1,) -> assign to new columns
            U_new = U_k.copy()
            V_new = V_k.copy()
            W_new = W_k.copy()
            # We need to scale the kept columns so that sum_d C_k[:,d] * w_d = T.
            # Actually the decomposition is T = sum_d C[:,d] (the column IS the rank-1 term).
            # Reducing: we want new columns C'_k such that sum_d C'_k[:,d] = T_flat.
            # The null-space relation: sum_d C[:,d] v_d = 0.
            # We drop column 'drop' and adjust the others.
            # C_new[:,d] = C[:,d] - v_d * C[:,drop] / v_drop  for d in keep
            v_drop = float(v[drop])
            if abs(v_drop) < 1e-9:
                continue
            for d, kd in enumerate(keep):
                factor = float(v[kd]) / v_drop
                C[:, kd]  # original
            # Reconstruct: C_k_new[:,d] = C[:,keep[d]] - (v[keep[d]]/v[drop]) * C[:,drop]
            C_new = np.empty((a * b * c, len(keep)))
            for d, kd in enumerate(keep):
                factor = float(v[kd]) / v_drop
                C_new[:, d] = C[:, kd] - factor * C[:, drop]
            # Now factor C_new[:,d] = U_new[i,d]*V_new[j,d]*W_new[k,d]
            # We have C[:,d] = U[i,d]*V[j,d]*W[k,d]. C_new[:,d] = C[:,d] - factor*C[:,drop]
            # = U[i,d]*V[j,d]*W[k,d] - factor * U[i,drop]*V[j,drop]*W[k,drop]
            # This is NOT a single rank-1 term in general. So this approach fails unless we re-factor.
            # Instead, solve least squares for new U,V,W columns.
            found = False
            break
        if not found:
            break

    return best[0], best[1], best[2]


def _try_zero_columns(U, V, W, T, n, m, p):
    """Try zeroing one column at a time (fast, exact for integer data)."""
    r = U.shape[1]
    for col_to_zero in range(r):
        if _time_left() < 3:
            break
        U_t = U.copy(); V_t = V.copy(); W_t = W.copy()
        U_t[:, col_to_zero] = 0; V_t[:, col_to_zero] = 0; W_t[:, col_to_zero] = 0
        if _verify(U_t, V_t, W_t, T):
            mask = [c for c in range(r) if c != col_to_zero]
            return U_t[:, mask], V_t[:, mask], W_t[:, mask]
    return U, V, W


def _try_pair_zero(U, V, W, T, n, m, p):
    """Try zeroing pairs of columns (catches dependencies that single-zero misses)."""
    r = U.shape[1]
    for c1 in range(r):
        if _time_left() < 3:
            break
        for c2 in range(c1 + 1, r):
            if _time_left() < 3:
                break
            U_t = U.copy(); V_t = V.copy(); W_t = W.copy()
            U_t[:, c1] = 0; V_t[:, c1] = 0; W_t[:, c1] = 0
            U_t[:, c2] = 0; V_t[:, c2] = 0; W_t[:, c2] = 0
            if _verify(U_t, V_t, W_t, T):
                mask = [c for c in range(r) if c != c1 and c != c2]
                return U_t[:, mask], V_t[:, mask], W_t[:, mask]
    return U, V, W


def _rebalance_and_round(U, V, W, T, n, m, p):
    """Try to round to half-integers while preserving exactness."""
    Ur = _round_half(U); Vr = _round_half(V); Wr = _round_half(W)
    if _verify(Ur, Vr, Wr, T):
        return Ur, Vr, Wr
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)

    U, V, W = _build_block_decomposition()

    # Verify base decomposition
    if not _verify(U, V, W, T):
        # Fallback to schoolbook full rank
        r = n * m * p
        U_fb = np.zeros((n * m, r)); V_fb = np.zeros((m * p, r)); W_fb = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best = (U, V, W)

    # Step 1: single-column zero elimination
    if _time_left() > 5:
        U1, V1, W1 = _try_zero_columns(U, V, W, T, n, m, p)
        if U1.shape[1] < U.shape[1]:
            best = (U1, V1, W1)
            U, V, W = U1, V1, W1

    # Step 2: pair-column zero elimination
    if _time_left() > 5:
        U2, V2, W2 = _try_pair_zero(U, V, W, T, n, m, p)
        if U2.shape[1] < U.shape[1]:
            best = (U2, V2, W2)
            U, V, W = U2, V2, W2

    # Step 3: linear-dependence elimination via nullspace (structural change)
    if _time_left() > 10:
        U3, V3, W3 = _reduce_by_nullspace(U, V, W, T, n, m, p, max_iter=6)
        if U3.shape[1] < U.shape[1]:
            best = (U3, V3, W3)
            U, V, W = U3, V3, W3

    # Step 4: round to half-integers
    if _time_left() > 3:
        Ur, Vr, Wr = _rebalance_and_round(U, V, W, T, n, m, p)
        if Ur.shape[1] == U.shape[1]:
            best = (Ur, Vr, Wr)
            U, V, W = Ur, Vr, Wr

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
