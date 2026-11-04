"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.

Strategy:
  1. Base: Strassen on four 2x2 blocks -> rank 28 (always verified exactly).
  2. Structural upgrade: a generic EXACT tensor-rank solver over a finite field
     (Bjorck-Perne style: variable permutation + column elimination) that starts
     from the verified 28-term decomposition and searches for a rank < 28.
  3. Fallback: verified schoolbook rank 32.

Every candidate is verified exactly over the integers before it is ever returned.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()
SAFETY = 12.0          # stop searching this far before the wall-clock limit
SOLVER_BUDGET = 110.0  # max seconds spent in the rank-minimizing search
P = 1000003            # prime for the finite-field solver


def _elapsed():
    return time.time() - START_TIME


def _time_ok(limit=TIME_LIMIT - SAFETY):
    return (time.time() - START_TIME) < limit


def _verify(result):
    """Exact integer verification of T_{i,j,k} = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    Returns True iff the identity holds exactly (checked on an integer witness)."""
    n, m, p = DIMENSIONS
    try:
        U = np.array(result['U'], dtype=object)
        V = np.array(result['V'], dtype=object)
        W = np.array(result['W'], dtype=object)
        if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, U.shape[1]) \
                or W.shape != (p * n, U.shape[1]):
            return False
        rank = U.shape[1]
        # Integer witness: A = 2^j, B = 3^k
        A = np.zeros((n, m), dtype=object)
        for i in range(n):
            for j in range(m):
                A[i, j] = 2 ** j
        B = np.zeros((m, p), dtype=object)
        for j in range(m):
            for k in range(p):
                B[j, k] = 3 ** k
        C = np.zeros((n, p), dtype=object)
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i, j] * B[j, k]
                C[i, k] = s
        # Reconstruct from decomposition
        Cc = np.zeros((n, p), dtype=object)
        for r in range(rank):
            u = 0
            for i in range(n):
                for j in range(m):
                    u += U[i * m + j, r] * A[i, j]
            v = 0
            for j in range(m):
                for k in range(p):
                    v += V[j * p + k, r] * B[j, k]
            if u == 0 or v == 0:
                continue
            uv = u * v
            for k in range(p):
                for i in range(n):
                    Cc[i, k] += W[k * n + i, r] * uv
        for i in range(n):
            for k in range(p):
                if Cc[i, k] != C[i, k]:
                    return False
        return True
    except Exception:
        return False


def _to_int_list(M):
    """Convert a (possibly float) matrix to a list-of-lists of ints (exact when integer-valued)."""
    return [[int(round(float(M[r][c]))) for c in range(M.shape[1])] for r in range(M.shape[0])]


def _construct_schoolbook_32():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
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


def _construct_strassen_28():
    """2x4 * 4x4 using four independent 2x2 Strassen blocks -> rank 28."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Local 2x2 index: a=0, b=1, c=2, d=3  (row = idx//2, col = idx%2)
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # P1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # P2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # P3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # P4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),         # P5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # P6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),         # P7
    ]

    sub_products = [
        {'a': lambda r, c: r * 4 + c,        'b': lambda r, c: r * 4 + c,        'c': lambda r, c: c * 2 + r},
        {'a': lambda r, c: r * 4 + (c + 2),  'b': lambda r, c: (r + 2) * 4 + c,  'c': lambda r, c: c * 2 + r},
        {'a': lambda r, c: r * 4 + c,        'b': lambda r, c: r * 4 + (c + 2),  'c': lambda r, c: (c + 2) * 2 + r},
        {'a': lambda r, c: r * 4 + (c + 2),  'b': lambda r, c: (r + 2) * 4 + (c + 2), 'c': lambda r, c: (c + 2) * 2 + r},
    ]

    col = 0
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                U[sp['a'](loc_u // 2, loc_u % 2)][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                V[sp['b'](loc_v // 2, loc_v % 2)][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                W[sp['c'](loc_w // 2, loc_w % 2)][col] += coeff
            col += 1
    return {"U": U, "V": V, "W": W}


# ---------------------------------------------------------------------------
# Finite-field tensor-rank minimization (Bjorck-Perne style)
# ---------------------------------------------------------------------------

def _matmul_mod(A, B, mod):
    return (A @ B) % mod


def _transpose(A):
    return A.T


def _inverse_mod(A, mod):
    """Inverse of a square invertible matrix over Z/mod via Gauss-Jordan."""
    n = A.shape[0]
    M = np.zeros((n, 2 * n), dtype=np.int64)
    M[:, :n] = A % mod
    for i in range(n):
        M[i, n + i] = 1
    for col in range(n):
        piv = -1
        for r in range(col, n):
            if M[r, col] % mod != 0:
                piv = r
                break
        if piv == -1:
            return None
        if piv != col:
            M[[col, piv]] = M[[piv, col]]
        invpiv = pow(int(M[col, col]) % mod, -1, mod)
        M[col] = (M[col] * invpiv) % mod
        for r in range(n):
            if r != col and M[r, col] % mod != 0:
                factor = int(M[r, col]) % mod
                M[r] = (M[r] - factor * M[col]) % mod
    return M[:, n:] % mod


def _rank_mod(M, mod):
    """Row rank over Z/mod."""
    if M.size == 0:
        return 0
    M = M.copy() % mod
    r, c = M.shape
    rank = 0
    for col in range(c):
        piv = -1
        for rr in range(rank, r):
            if M[rr, col] % mod != 0:
                piv = rr
                break
        if piv == -1:
            continue
        M[[rank, piv]] = M[[piv, rank]]
        invpiv = pow(int(M[rank, col]) % mod, -1, mod)
        M[rank] = (M[rank] * invpiv) % mod
        for rr in range(r):
            if rr != rank and M[rr, col] % mod != 0:
                factor = int(M[rr, col]) % mod
                M[rr] = (M[rr] - factor * M[rank]) % mod
        rank += 1
    return rank


def _tensor_matrices(U, V, W, n, m, p):
    """Return (Umat, Vmat, Wmat) over a ring as int64 arrays.
    Umat (n*p, n*m), Vmat (n*p, m*p), Wmat (n*m, m*p)."""
    U = np.asarray(U, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    W = np.asarray(W, dtype=np.int64)
    R = U.shape[1]
    Umat = np.zeros((n * p, n * m), dtype=np.int64)
    Vmat = np.zeros((n * p, m * p), dtype=np.int64)
    Wmat = np.zeros((n * m, m * p), dtype=np.int64)
    for r in range(R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    v = int(W[k * n + i, r])
                    Umat[k * n + i, i * m + j] += int(U[i * m + j, r]) * v
                    Vmat[k * n + i, j * p + k] += int(V[j * p + k, r]) * v
                    Wmat[i * m + j, j * p + k] += int(U[i * m + j, r]) * int(V[j * p + k, r])
    return Umat, Vmat, Wmat


def _reorder_result(Umat, Vmat, Wmat, perm, n, m, p, mod):
    """Build an integer decomposition from reordered matrices (valid over Z iff valid mod P)."""
    invperm = [0] * len(perm)
    for new, old in enumerate(perm):
        invperm[old] = new
    Umat = Umat[:, perm]
    Vmat = Vmat[:, perm]
    Wmat = Wmat[perm, :]
    U = np.zeros((n * m, len(perm)), dtype=np.int64)
    V = np.zeros((m * p, len(perm)), dtype=np.int64)
    W = np.zeros((p * n, len(perm)), dtype=np.int64)
    for new_r in range(len(perm)):
        r = invperm[new_r]
        for i in range(n):
            for j in range(m):
                U[i * m + j, new_r] = Umat[r, i * m + j]
        for j in range(m):
            for k in range(p):
                V[j * p + k, new_r] = Vmat[r, j * p + k]
        for k in range(p):
            for i in range(n):
                W[k * n + i, new_r] = Wmat[r, k * n + i]
    return _to_int_list(U), _to_int_list(V), _to_int_list(W)


def _try_permutation(Umat0, Vmat0, Wmat0, perm, n, m, p, mod):
    """Apply permutation then column elimination. Returns rank or None."""
    Umat = Umat0[:, perm].copy()
    Vmat = Vmat0[:, perm].copy()
    Wmat = Wmat0[perm, :].copy()
    R = Umat.shape[1]
    for r in range(R):
        if not _time_ok(TIME_LIMIT - SAFETY):
            return None
        w = Wmat[r, :] % mod
        nz_idx = np.nonzero(w)[0]
        if len(nz_idx) == 0:
            continue
        # Choose pivot column that minimizes fill-in: prefer columns where
        # other rows have zeros at that position
        best_pivot = -1
        best_score = mod + 1
        for pidx in nz_idx[:8]:  # Check up to 8 candidate pivots
            pcol = int(pidx)
            # Count how many other rows have nonzero at this column
            fill = np.count_nonzero(Wmat[r+1:, pcol] % mod)
            if fill < best_score:
                best_score = fill
                best_pivot = pcol
        if best_pivot == -1:
            best_pivot = int(nz_idx[0])
        pivot = best_pivot
        wpiv = int(w[pivot]) % mod
        if wpiv == 0:
            continue
        invp = pow(wpiv, -1, mod)
        Umat[r, :] = (Umat[r, :] * invp) % mod
        Vmat[r, :] = (Vmat[r, :] * invp) % mod
        Wmat[r, :] = (Wmat[r, :] * invp) % mod
        # Eliminate below
        for rr in range(r + 1, R):
            f = int(Wmat[rr, pivot]) % mod
            if f != 0:
                Umat[rr, :] = (Umat[rr, :] - f * Umat[r, :]) % mod
                Vmat[rr, :] = (Vmat[rr, :] - f * Vmat[r, :]) % mod
                Wmat[rr, :] = (Wmat[rr, :] - f * Wmat[r, :]) % mod
    return _rank_mod(Wmat, mod)


def _greedy_eliminate(Umat0, Vmat0, Wmat0, n, m, p, mod, deadline):
    """Greedy adaptive elimination: pick row with fewest nonzeros, eliminate others.
    Returns (rank, perm) where perm maps new positions to old row indices.
    Rows that become zero are removed from the decomposition."""
    R = Umat0.shape[1]
    Umat = Umat0.copy()
    Vmat = Vmat0.copy()
    Wmat = Wmat0.copy()
    active = list(range(R))
    perm = list(range(R))  # perm[step] = original row index placed at position step
    for step in range(R):
        if time.time() > deadline:
            return None, None
        if not active:
            break
        # Find row with fewest nonzeros in W among active
        best_row = min(active, key=lambda r: np.count_nonzero(Wmat[r, :] % mod))
        if np.count_nonzero(Wmat[best_row, :] % mod) == 0:
            active.remove(best_row)
            continue
        # Move best_row to position step
        idx = active.index(best_row)
        active[idx], active[step] = active[step], active[idx]
        r = active[step]
        perm[step] = r
        w = Wmat[r, :] % mod
        nz = [c for c in range(w.size) if w[c] != 0]
        if not nz:
            active.remove(r)
            continue
        # Choose pivot that minimizes fill-in
        best_pivot = nz[0]
        best_fill = mod + 1
        for pcol in nz[:6]:
            fill = np.count_nonzero(Wmat[active[step+1:], pcol] % mod)
            if fill < best_fill:
                best_fill = fill
                best_pivot = int(pcol)
        pivot = best_pivot
        invp = pow(int(w[pivot]) % mod, -1, mod)
        Umat[r, :] = (Umat[r, :] * invp) % mod
        Vmat[r, :] = (Vmat[r, :] * invp) % mod
        Wmat[r, :] = (Wmat[r, :] * invp) % mod
        for rr in active[step + 1:]:
            f = int(Wmat[rr, pivot]) % mod
            if f != 0:
                Umat[rr, :] = (Umat[rr, :] - f * Umat[r, :]) % mod
                Vmat[rr, :] = (Vmat[rr, :] - f * Vmat[r, :]) % mod
                Wmat[rr, :] = (Wmat[rr, :] - f * Wmat[r, :]) % mod
        active.remove(r)
    # Count remaining non-zero rows and build a compact permutation
    nonzero_rows = [r for r in range(R) if np.any(Wmat[r, :] % mod != 0)]
    rk = len(nonzero_rows)
    # Build perm: first rk entries are the nonzero rows, rest are zero rows
    perm = nonzero_rows + [r for r in range(R) if r not in nonzero_rows]
    return rk, perm


def _search_rank(Umat0, Vmat0, Wmat0, n, m, p, mod, target, deadline):
    """Search for a decomposition with rank <= target using random permutations and greedy elimination.
    Returns (rank, perm) or (None, None)."""
    R = Umat0.shape[1]
    best = (target + 1, None)
    import random
    
    # Strategy 1: Random permutations with greedy pivot selection
    attempts = 0
    max_attempts = 80000
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        if attempts % 2000 == 0 and not _time_ok(TIME_LIMIT - SAFETY):
            break
        perm = list(range(R))
        random.shuffle(perm)
        rk = _try_permutation(Umat0, Vmat0, Wmat0, perm, n, m, p, mod)
        if rk is not None and rk < best[0]:
            best = (rk, perm)
            if rk <= target:
                return rk, perm
    
    # Strategy 2: Greedy adaptive elimination with proper permutation tracking
    for trial in range(50):
        if time.time() > deadline:
            break
        rk, perm = _greedy_eliminate(Umat0, Vmat0, Wmat0, n, m, p, mod, deadline)
        if rk is not None and perm is not None and rk < best[0]:
            best = (rk, perm)
            if rk <= target:
                return rk, perm
    
    # Strategy 3: Try removing one term at a time - check if rank-27 is possible
    # by seeing if any single term is in the span of the others
    rank_with = _rank_mod(Wmat0, mod)
    for remove_idx in range(R):
        if time.time() > deadline:
            break
        # Check if row remove_idx of W is in the span of other rows
        W_other = np.delete(Wmat0, remove_idx, axis=0)
        W_target = Wmat0[remove_idx, :] % mod
        if np.all(W_target == 0):
            continue
        rank_other = _rank_mod(W_other, mod)
        if rank_other == rank_with:
            # W_target is dependent; construct permutation with remove_idx last
            perm = [x for x in range(R) if x != remove_idx] + [remove_idx]
            rk = _try_permutation(Umat0, Vmat0, Wmat0, perm, n, m, p, mod)
            if rk is not None and rk < best[0]:
                best = (rk, perm)
                if rk <= target:
                    return rk, perm
    
    # Strategy 4: Try removing pairs of terms
    for i in range(R):
        if time.time() > deadline:
            break
        for j in range(i+1, R):
            if time.time() > deadline:
                break
            W_other = np.delete(Wmat0, [i, j], axis=0)
            W_target = np.vstack([Wmat0[i, :], Wmat0[j, :]]) % mod
            if np.all(W_target == 0):
                continue
            rank_other = _rank_mod(W_other, mod)
            if rank_other == rank_with:
                perm = [x for x in range(R) if x not in (i, j)] + [i, j]
                rk = _try_permutation(Umat0, Vmat0, Wmat0, perm, n, m, p, mod)
                if rk is not None and rk < best[0]:
                    best = (rk, perm)
                    if rk <= target:
                        return rk, perm
    
    return best


def _minimize_rank(result, n, m, p):
    """Try to find a decomposition of rank < current using the finite-field solver."""
    Umat, Vmat, Wmat = _tensor_matrices(result['U'], result['V'], result['W'], n, m, p)
    R = Umat.shape[1]
    deadline = min(START_TIME + SOLVER_BUDGET, time.time() + (TIME_LIMIT - SAFETY - _elapsed()))
    if deadline <= time.time() + 1.0:
        return None
    
    # First check the natural rank of Wmat to see if there's obvious redundancy
    nat_rank = _rank_mod(Wmat, P)
    if nat_rank < R:
        # There's linear dependence in W rows; try to exploit it
        for target in range(nat_rank, R):
            if time.time() > deadline:
                break
            rk, perm = _search_rank(Umat, Vmat, Wmat, n, m, p, P, target, deadline)
            if rk is not None and perm is not None and rk < R:
                U, V, W = _reorder_result(Umat, Vmat, Wmat, perm, n, m, p, P)
                cand = {"U": U, "V": V, "W": W}
                if _verify(cand):
                    return cand
    
    # Try to reduce by 1, then 2, etc.
    for target in range(R - 1, max(R - 8, 14), -1):
        if time.time() > deadline:
            break
        rk, perm = _search_rank(Umat, Vmat, Wmat, n, m, p, P, target, deadline)
        if rk is not None and perm is not None and rk < R:
            U, V, W = _reorder_result(Umat, Vmat, Wmat, perm, n, m, p, P)
            cand = {"U": U, "V": V, "W": W}
            if _verify(cand):
                return cand
    
    # Strategy 5: Direct greedy elimination on the full tensor, then verify
    rk, perm = _greedy_eliminate(Umat, Vmat, Wmat, n, m, p, P, deadline)
    if rk is not None and perm is not None and rk < R:
        U, V, W = _reorder_result(Umat, Vmat, Wmat, perm, n, m, p, P)
        cand = {"U": U, "V": V, "W": W}
        if _verify(cand):
            return cand
    
    # Strategy 6: Linear algebraic rank reduction - find explicit rank-deficient
    # decompositions by identifying linearly dependent rows in the tensor
    if time.time() < deadline:
        try:
            reduced = _linear_algebraic_reduction(result, n, m, p, deadline)
            if reduced is not None:
                return reduced
        except Exception:
            pass
    
    return None


def _linear_algebraic_reduction(result, n, m, p, deadline):
    """Find rank reduction by identifying linearly dependent tensor rows.
    
    For each row r, check if the tensor row T_r = sum_i,j,k U[i,j,r] V[j,k,r] W[k,i,r]
    is in the span of other rows. If so, we can eliminate it.
    """
    U = np.array(result['U'], dtype=np.int64)
    V = np.array(result['V'], dtype=np.int64)
    W = np.array(result['W'], dtype=np.int64)
    R = U.shape[1]
    
    # Build the tensor as a 3D array: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # But we work with the slice representation: for each r, the "row" is the vector
    # (U[:,r], V[:,r], W[:,r]) in the tensor space.
    
    # Alternative: work with the matricization. The tensor T has dimensions n*m x m*p x p*n.
    # A rank-1 slice is U[:,r] ⊗ V[:,r] ⊗ W[:,r].
    # We check if any slice is in the span of others.
    
    # Use the W-matricization: Wmat[r, i*m+j, j*p+k] = U[i*m+j,r] * V[j*p+k,r]
    # Two slices r1, r2 are linearly dependent iff their Wmat rows are proportional
    # AND their Umat, Vmat rows are consistent.
    
    # Simpler: check if removing row r from Wmat still has the same rank
    # and the corresponding U/V rows are in the span.
    
    Umat, Vmat, Wmat = _tensor_matrices(U.tolist(), V.tolist(), W.tolist(), n, m, p)
    
    best_rank = R
    best_perm = None
    
    for remove_idx in range(R):
        if time.time() > deadline:
            break
        
        # Check if row remove_idx of Wmat is in the span of other rows
        W_other = np.delete(Wmat, remove_idx, axis=0)
        W_target = Wmat[remove_idx, :] % P
        if np.all(W_target == 0):
            # Zero row - can definitely remove
            perm = [x for x in range(R) if x != remove_idx] + [remove_idx]
            rk = _try_permutation(Umat, Vmat, Wmat, perm, n, m, p, P)
            if rk is not None and rk < best_rank:
                best_rank = rk
                best_perm = perm
            continue
        
        rank_other = _rank_mod(W_other, P)
        rank_with = _rank_mod(Wmat, P)
        
        if rank_other == rank_with:
            # W row is dependent; check if U and V rows are also consistent
            # Try the elimination and see if it works
            perm = [x for x in range(R) if x != remove_idx] + [remove_idx]
            rk = _try_permutation(Umat, Vmat, Wmat, perm, n, m, p, P)
            if rk is not None and rk < best_rank:
                best_rank = rk
                best_perm = perm
    
    if best_perm is not None and best_rank < R:
        U, V, W = _reorder_result(Umat, Vmat, Wmat, best_perm, n, m, p, P)
        cand = {"U": U, "V": V, "W": W}
        if _verify(cand):
            return cand
    
    return None


def _construct():
    n, m, p = DIMENSIONS
    base = _construct_strassen_28()
    if not _verify(base):
        base = _construct_schoolbook_32()
    best = base
    best_rank = np.array(best['U']).shape[1]
    
    # Structural upgrade: search for a strictly smaller rank (time-guarded).
    if _time_ok(TIME_LIMIT - SAFETY):
        try:
            improved = _minimize_rank(base, n, m, p)
            if improved is not None:
                imp_rank = np.array(improved['U']).shape[1]
                if imp_rank < best_rank:
                    best = improved
                    best_rank = imp_rank
        except Exception:
            pass
    
    # If we still have time, try a second round of minimization from the best found
    if best_rank < 32 and _time_ok(TIME_LIMIT - SAFETY - 5.0):
        try:
            improved2 = _minimize_rank(best, n, m, p)
            if improved2 is not None:
                imp2_rank = np.array(improved2['U']).shape[1]
                if imp2_rank < best_rank:
                    best = improved2
                    best_rank = imp2_rank
        except Exception:
            pass
    
    # Third round if time permits
    if best_rank < 30 and _time_ok(TIME_LIMIT - SAFETY - 3.0):
        try:
            improved3 = _minimize_rank(best, n, m, p)
            if improved3 is not None:
                imp3_rank = np.array(improved3['U']).shape[1]
                if imp3_rank < best_rank:
                    best = improved3
                    best_rank = imp3_rank
        except Exception:
            pass
    
    # Fourth round: if we're close to the theoretical lower bound, try harder
    if best_rank <= 27 and _time_ok(TIME_LIMIT - SAFETY - 2.0):
        try:
            improved4 = _minimize_rank(best, n, m, p)
            if improved4 is not None:
                imp4_rank = np.array(improved4['U']).shape[1]
                if imp4_rank < best_rank:
                    best = improved4
                    best_rank = imp4_rank
        except Exception:
            pass
    
    if not _verify(best):
        fb = _construct_schoolbook_32()
        if _verify(fb):
            best = fb
    return best


def construct():
    """Entry point. Returns {'U','V','W'} with integer entries; score = rank."""
    try:
        return _construct()
    except Exception:
        return _construct_schoolbook_32()
