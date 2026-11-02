"""Fresh design: randomized structured search for a rank-minimal exact tensor
decomposition of the (2,3,5) matrix-multiplication tensor, with exact
Fraction verification and a wall-clock deadline guard."""

import time
import random
import fractions

DIMENSIONS = [2, 3, 5]

_START = time.time()
_DEADLINE = _START + 130.0  # 20 s safety margin before the 150 s kill


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W, n, m, p):
    """Return True iff sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k."""
    rank = len(U[0])
    F = fractions.Fraction
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = F(0)
                w = F(W[k * n + i][0])
                for c in range(rank):
                    s += F(U[i * m + j][c]) * F(V[j * p + k][c])
                    # accumulate W contribution separately to keep it exact
                # recompute: s should equal W
                # (W is constant per (i,k), independent of j)
                # We'll do a clean check below instead.
    # Clean exact check:
    F = fractions.Fraction
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = F(0)
                for c in range(rank):
                    s += F(U[i * m + j][c]) * F(V[j * p + k][c])
                if s != F(W[k * n + i][0]):
                    return False
    return True


def _to_halfint(x):
    """Round a float to the nearest half-integer (multiple of 0.5)."""
    return round(x * 2.0) / 2.0


def _rand_small(nrows, ncols, scale=2.0, p_nonzero=0.6, rng=None):
    rng = rng or random
    M = []
    for _ in range(nrows):
        row = []
        for _ in range(ncols):
            if rng.random() < p_nonzero:
                v = rng.randint(-scale, scale) / 2.0
                row.append(v)
            else:
                row.append(0.0)
        M.append(row)
    return M


def _build_identity_linear_constraints(n, m, p, rank, U, V):
    """Return list of (coeff_dict, rhs) where coeff_dict maps column c to coeff,
    such that sum_c coeff[c]*W[k*n+i,c] = rhs for each (i,j,k)."""
    F = fractions.Fraction
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                coeffs = {}
                for c in range(rank):
                    val = F(U[i * m + j][c]) * F(V[j * p + k][c])
                    if val != 0:
                        coeffs[c] = coeffs.get(c, F(0)) + val
                rhs = F(1)  # W[k*n+i] must equal 1
                constraints.append((coeffs, rhs))
    return constraints


def _solve_for_W(constraints, rank, n, p):
    """Given constraints on W (a vector of length p*n indexed by k*n+i),
    find a W satisfying all. Uses Gaussian elimination over Fractions.
    Returns W as list of Fractions, or None if inconsistent."""
    F = fractions.Fraction
    num_vars = p * n
    # Build augmented system: A (num_constraints x num_vars) | b
    rows = []
    for coeffs, rhs in constraints:
        row = [F(0)] * num_vars
        for c, val in coeffs.items():
            row[c] = val
        rows.append(row + [rhs])

    # Gaussian elimination
    nrows = len(rows)
    col_used = [False] * num_vars
    rank_mat = 0
    pivot_row = 0
    for col in range(num_vars):
        # find pivot
        piv = None
        for r in range(pivot_row, nrows):
            if rows[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        rows[pivot_row], rows[piv] = rows[piv], rows[pivot_row]
        # normalize
        pv = rows[pivot_row][col]
        for c in range(col, num_vars + 1):
            rows[pivot_row][c] = rows[pivot_row][c] / pv
        # eliminate
        for r in range(nrows):
            if r != pivot_row and rows[r][col] != 0:
                factor = rows[r][col]
                for c in range(col, num_vars + 1):
                    rows[r][c] -= factor * rows[pivot_row][c]
        col_used[col] = True
        rank_mat += 1
        pivot_row += 1
        if pivot_row == nrows:
            break

    # check consistency: rows that are all-zero in A must have 0 rhs
    for r in range(nrows):
        allzero = all(rows[r][c] == 0 for c in range(num_vars))
        if allzero and rows[r][num_vars] != 0:
            return None

    # free variables set to 0; basic variables read off
    W = [F(0)] * num_vars
    for col in range(num_vars):
        if not col_used[col]:
            W[col] = F(0)
    # recompute basic vars from rows (each pivot row has leading 1 at its col)
    # We need pivot columns per row. Re-derive:
    pivot_cols = []
    for r in range(nrows):
        for c in range(num_vars):
            if rows[r][c] != 0:
                pivot_cols.append((r, c))
                break
    # Solve: for each pivot row r with pivot col c, W[c] = rhs - sum_{c2>c} A[r][c2]*W[c2]
    # Process columns in reverse
    for r, c in reversed(pivot_cols):
        val = rows[r][num_vars]
        for c2 in range(c + 1, num_vars):
            val -= rows[r][c2] * W[c2]
        W[c] = val
    return W


def _search(n, m, p, deadline, seed):
    rng = random.Random(seed)
    best = None
    # Try ranks below n*m*p
    for rank in range(1, n * m * p):
        if time.time() > deadline:
            break
        # number of trials budgeted by time
        trials = 0
        while time.time() < deadline:
            trials += 1
            U = _rand_small(n * m, rank, scale=2, p_nonzero=0.7, rng=rng)
            V = _rand_small(m * p, rank, scale=2, p_nonzero=0.7, rng=rng)
            W = _rand_small(p * n, rank, scale=2, p_nonzero=0.7, rng=rng)
            if _verify_exact(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W, "rank": rank}
                return best
    return best


def construct():
    n, m, p = DIMENSIONS
    # Fallback: schoolbook (always valid, rank 30)
    fallback = _schoolbook(n, m, p)

    # Try to find a smaller exact rank via randomized search.
    best = None
    for seed in range(0, 40):
        if time.time() > _DEADLINE:
            break
        res = _search(n, m, p, _DEADLINE, seed)
        if res is not None:
            if best is None or res["rank"] < best["rank"]:
                best = res
            if best["rank"] <= 1:
                break

    if best is None:
        return fallback

    # Return integer/half-integer matrices (they already are)
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
