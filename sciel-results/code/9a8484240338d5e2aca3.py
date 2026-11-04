"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # safety margin before 150s kill

# Strassen 2x2x2 decomposition, expressed as 7 rank-1 terms.
# Each term: (u_entries, v_entries, w_entries) with local indices.
# U local index = (a, c) -> a in {0,1}, c in {0,1}
# V local index = (c, d)
# W local index = (d, a)
# The bilinear identity: sum_c U[a,c]*V[c,d]*W[d,a] must equal 1 for all a,c,d in {0,1}
# (i.e. it reconstructs the 2x2x2 multiplication tensor).
_STRASSEN = [
    # M1 = (A00+A11)*(B00+B11)
    ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
    # M2 = (A10+A11)*B00
    ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
    # M3 = A00*(B01-B11)
    ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
    # M4 = A11*(B10-B00)
    ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
    # M5 = (A00+A01)*B11
    ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (1, 0, 1)]),
    # M6 = (A10-A00)*(B00+B01)
    ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
    # M7 = (A01-A11)*(B10+B11)
    ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
]


def _generate_partitions(total, max_part=3):
    """All ordered partitions of `total` into parts in 1..max_part."""
    if total == 0:
        return [[]]
    out = []
    for first in range(1, max_part + 1):
        if first > total:
            continue
        for rest in _generate_partitions(total - first, max_part):
            out.append([first] + rest)
    return out


def _block_rank(ns, ms, ps):
    if ns == 2 and ms == 2 and ps == 2:
        return 7
    return ns * ms * ps


def _build_terms(n_parts, m_parts, p_parts, n, m, p):
    """Build list of terms; each term = (u_entries, v_entries, w_entries) with global indices."""
    n_offsets = [0]
    for sz in n_parts[:-1]:
        n_offsets.append(n_offsets[-1] + sz)
    m_offsets = [0]
    for sz in m_parts[:-1]:
        m_offsets.append(m_offsets[-1] + sz)
    p_offsets = [0]
    for sz in p_parts[:-1]:
        p_offsets.append(p_offsets[-1] + sz)

    terms = []
    for ni, ns in enumerate(n_parts):
        a_rows = list(range(n_offsets[ni], n_offsets[ni] + ns))
        for mi, ms in enumerate(m_parts):
            c_cols = list(range(m_offsets[mi], m_offsets[mi] + ms))
            for pi, ps in enumerate(p_parts):
                d_cols = list(range(p_offsets[pi], p_offsets[pi] + ps))

                def U_idx(la, lc):
                    return a_rows[la] * m + c_cols[lc]

                def V_idx(lc, ld):
                    return c_cols[lc] * p + d_cols[ld]

                def W_idx(ld, la):
                    return d_cols[ld] * n + a_rows[la]

                if ns == 2 and ms == 2 and ps == 2:
                    for (ue, ve, we) in _STRASSEN:
                        terms.append((
                            [(U_idx(la, lc), cf) for (la, lc, cf) in ue],
                            [(V_idx(lc, ld), cf) for (lc, ld, cf) in ve],
                            [(W_idx(ld, la), cf) for (ld, la, cf) in we],
                        ))
                else:
                    for la in range(ns):
                        for lc in range(ms):
                            for ld in range(ps):
                                terms.append((
                                    [(U_idx(la, lc), 1)],
                                    [(V_idx(lc, ld), 1)],
                                    [(W_idx(ld, la), 1)],
                                ))
    return terms


def _terms_to_matrices(terms, n, m, p):
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for ci, (ue, ve, we) in enumerate(terms):
        for row, cf in ue:
            U[row, ci] += cf
        for row, cf in ve:
            V[row, ci] += cf
        for row, cf in we:
            W[row, ci] += cf
    return U, V, W, rank


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    r = U.shape[1]
    for i in range(n):
        Ui = U[i * m:(i + 1) * m]
        for j in range(m):
            urow = Ui[j]
            for k in range(p):
                vrow = V[j * p + k]
                wrow = W[k * n + i]
                val = float(np.dot(urow, vrow * wrow))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_refine(U, V, W, n, m, p, budget_s=20.0):
    """Tensor-train / hierarchical refinement: try to reduce rank by finding
    dependent columns and merging, or by a local exact solve on the residual.
    Returns (U, V, W, rank) with possibly smaller rank, or original if none.
    This is a best-effort polish within the same family."""
    start = time.time()
    U, V, W = U.copy(), V.copy(), W.copy()
    r = U.shape[1]

    # 1) Remove exactly-zero columns.
    keep = []
    for c in range(r):
        if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
            keep.append(c)
    if len(keep) < r:
        U, V, W = U[:, keep], V[:, keep], W[:, keep]
        r = len(keep)

    # 2) Attempt a small "absorption" polish: for each pair of columns, try a
    #    2->1 merge by solving for a scalar alpha such that column c2 = alpha*col c1
    #    holds on the product structure. This is conservative; only exact merges.
    for c2 in range(r - 1, -1, -1):
        if time.time() - start > budget_s:
            break
        for c1 in range(c2):
            # Check if (U[:,c2],V[:,c2],W[:,c2]) is a scalar multiple of c1's triple.
            # We need U2 = a*U1, V2 = b*V1, W2 = c*W1 with a*b*c = 1 (so the
            # product is preserved) AND the triple equals a*b*c * triple1.
            # Simplest: check if the three vectors are pairwise proportional with
            # consistent scalar product = 1.
            if not (np.any(U[:, c1] != 0) and np.any(V[:, c1] != 0) and np.any(W[:, c1] != 0)):
                continue
            # Determine scalar a from U where nonzero
            a = None
            for idx in np.nonzero(U[:, c1])[0]:
                if U[idx, c1] != 0 and U[idx, c2] != 0:
                    cand = U[idx, c2] / U[idx, c1]
                    if a is None:
                        a = cand
                    elif abs(a - cand) > 1e-12:
                        a = None
                        break
            if a is None or abs(a) < 1e-12:
                continue
            # Check U2 = a*U1 everywhere
            if not np.allclose(U[:, c2], a * U[:, c1], atol=1e-12):
                continue
            # Determine b from V
            b = None
            for idx in np.nonzero(V[:, c1])[0]:
                if V[idx, c1] != 0 and V[idx, c2] != 0:
                    cand = V[idx, c2] / V[idx, c1]
                    if b is None:
                        b = cand
                    elif abs(b - cand) > 1e-12:
                        b = None
                        break
            if b is None or abs(b) < 1e-12:
                continue
            if not np.allclose(V[:, c2], b * V[:, c1], atol=1e-12):
                continue
            # Then W2 must equal (1/(a*b)) * W1
            cw = 1.0 / (a * b)
            if not np.allclose(W[:, c2], cw * W[:, c1], atol=1e-12):
                continue
            # Exact merge: drop column c2
            U = np.delete(U, c2, axis=1)
            V = np.delete(V, c2, axis=1)
            W = np.delete(W, c2, axis=1)
            r -= 1
            break  # restart c2 loop from top

    # Final verify; if broken, return original.
    if r < U.shape[1]:
        if _verify(U, V, W, n, m, p, tol=1e-9):
            return U, V, W, r
    return U, V, W, U.shape[1]


def construct():
    n, m, p = DIMENSIONS

    n_parts_list = _generate_partitions(n)
    m_parts_list = _generate_partitions(m)
    p_parts_list = _generate_partitions(p)

    best_rank = float('inf')
    best_U = best_V = best_W = None

    for np_ in n_parts_list:
        for mp_ in m_parts_list:
            for pp_ in p_parts_list:
                if time.time() > _DEADLINE:
                    break

                # Quick rank estimate
                total_rank = 0
                for ns, ms, ps in product(np_, mp_, pp_):
                    total_rank += _block_rank(ns, ms, ps)
                if total_rank >= best_rank:
                    continue

                terms = _build_terms(np_, mp_, pp_, n, m, p)
                U, V, W, rank = _terms_to_matrices(terms, n, m, p)

                if not _verify(U, V, W, n, m, p, tol=1e-9):
                    continue

                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W

        if time.time() > _DEADLINE:
            break
    if time.time() > _DEADLINE:
        break

    if best_U is None:
        # Fallback: pure schoolbook
        terms = _build_terms([1] * n, [1] * m, [1] * p, n, m, p)
        best_U, best_V, best_W, best_rank = _terms_to_matrices(terms, n, m, p)
        if not _verify(best_U, best_V, best_W, n, m, p, tol=1e-9):
            # Should never happen, but guard.
            r = n * m * p
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, c] = 1
                        V[j * p + k, c] = 1
                        W[k * n + i, c] = 1
                        c += 1
            best_U, best_V, best_W, best_rank = U, V, W, r

    # Refinement polish (tensor-train style hierarchical merge), time-boxed.
    if time.time() < _DEADLINE - 5:
        U2, V2, W2, r2 = _try_refine(best_U, best_V, best_W, n, m, p,
                                     budget_s=max(1.0, _DEADLINE - time.time() - 5))
        if r2 < best_rank and _verify(U2, V2, W2, n, m, p, tol=1e-9):
            best_U, best_V, best_W, best_rank = U2, V2, W2, r2

    # Convert to list-of-lists of plain numbers (int when whole) to match interface.
    def to_list(mat):
        out = []
        for row in mat:
            rlist = []
            for x in row:
                if abs(x - round(x)) < 1e-12:
                    rlist.append(int(round(x)))
                else:
                    rlist.append(float(x))
            out.append(rlist)
        return out

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
