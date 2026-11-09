"""Global exact bilinear rank search for the 2x3x5 matrix-multiplication tensor.

Strategy (structurally different from the seed's incremental one-term-removal):
  * For a candidate rank r, generate a pool of "patterns" (support masks) for the
    three factor matrices. Patterns include Strassen-2x2x2 templates, shifted
    Strassen, schoolbook, random sparse masks, and the all-ones mask.
  * For each assignment of r patterns (with repetition, capped), build the
    bilinear system:  U[i,j,a] * V[j,k,a] * W[k,i,a]  summed over a must equal 1
    for every (i,j,k).  This is bilinear in the unknowns.
  * Solve exactly over Q by fixing all variables except one (the last) and
    substituting: the equation becomes  linear_in_last + quadratic_in_last * 0
    because each equation is linear in the last variable given the others are
    fixed.  Wait -- it's NOT linear in the last variable because U*V*W is
    bilinear in pairs.  Actually, given all variables in terms 1..r-1 are fixed,
    term r contributes U_r[i,j] * V_r[j,k] * W_r[k,i] which is a product of
    three unknowns -- still multilinear.  So fixing all but one variable does
    NOT make it linear.

  Correct approach: use the fact that the system is multilinear.  We solve it by
  *iterative substitution*: pick a variable, and for each equation it appears in,
  the equation is linear in that variable (since the other two factors in that
    product are other variables, but if we fix ALL other variables first, then
    each equation is linear in the target variable).  So:
    1. Assign values to all variables except the last one (x_last).
    2. Each equation becomes: (known sum) + x_last * (coefficient) = 1
       where coefficient = product of the other two factors in term r for that
       (i,j,k).  This is linear in x_last.
    3. Solve for x_last from one equation, verify in all others.
  But this only works if we can choose the other variables freely.  The problem
  is we have 30 equations and up to 3*r unknowns.  For r < 10, 3r < 30, so the
  system is overdetermined and we can't freely assign.

  Better: use the *rank-1 update* structure.  The tensor T = sum_a u_a v_a^T w_a
  where u_a is in R^{6}, v_a in R^{15}, w_a in R^{10}.  This is a 3-tensor
  decomposition.  For small r, we can do a *grid search* over rational values
  in a small set, but that's exponential.

  Practical approach for this specific small case:
  - The known lower bound for 2x3x5 is 23 (from literature).
  - The current program achieves 29.
  - We'll try ranks from 23 upward and for each rank, do a randomized search:
    initialize U, V, W with random rational entries (small numerators/denominators),
    then use a *projected gradient* or *Newton* step on the residual
    R = T - sum_a U[:,a] V[:,a] W[:,a]  (elementwise).
    But exact rational Newton is hard.

  Alternative practical approach: since the problem is tiny (30 equations),
  use the *border rank* / *catalecticant* approach?  Too complex.

  Most practical: for each rank r from 23 to 30, try to find a decomposition by
  solving the system of 30 polynomial equations in 3r unknowns over Q using a
  *Groebner basis* or *resultant* approach.  For r <= 10, 3r <= 30, so the
  system is square or nearly square.  We can use the fact that each equation is
  multilinear (degree 3 total, but degree 1 in each factor-group).

  KEY INSIGHT: Fix the values of U and V (6r + 15r = 21r unknowns).  Then each
  equation is LINEAR in the W variables (10r unknowns).  So:
    1. For a given (U, V), solve the linear system in W.
    2. The condition that this linear system has a solution is that the RHS
       is in the column space.  This gives constraints on (U, V).
  But choosing (U, V) is still hard.

  PRACTICAL IMPLEMENTATION: Use a *randomized* approach with exact arithmetic:
  - For rank r, randomly assign rational values to U and V (from a small set
    like {-1, 0, 1, 1/2}).
  - Solve the linear system for W over Q.
  - If a solution exists, verify exactly.  If yes, return.
  - Repeat with different random assignments.
  This is a Monte Carlo method but with exact arithmetic, so any hit is valid.
  For r close to the true rank, the probability of hitting a valid assignment
  with small rationals may be low, but for r = 29 (known feasible) it should
  find one quickly.  For r = 23-28, it may or may not find one within the
  time budget, but it's worth trying.

  To improve the odds: use *structured* initialization:
  - Start from the known rank-29 decomposition and randomly perturb.
  - Use the Strassen pattern as a starting point.
"""
from fractions import Fraction
import random
import time
from itertools import product

DIMENSIONS = [2, 3, 5]


def _half(M):
    out = []
    for row in M:
        r = []
        for v in row:
            if isinstance(v, Fraction):
                f = v * 2
                if f.denominator == 1:
                    fv = f.numerator
                    r.append(fv // 2 if fv % 2 == 0 else fv / 2)
                else:
                    r.append(float(v))
            else:
                r.append(v)
        out.append(row)
        out.append(r)
    return out


def _check(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                for c in range(rank):
                    a = uc[c]
                    if a:
                        s += a * vc[c] * wc[c]
                if s != 1:
                    return False
    return True


def _build_rank29(n, m, p):
    """Strassen 2x2x2 on (i in {0,1}, j in {0,1}, k in {0,1}) + schoolbook rest."""
    bini = [
        (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
        (0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0),
        (1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0),
        (0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1),
        (1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0),
        (1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0),
    ]
    rank = 29
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for (u00, u01, u10, u11, v00, v01, v10, v11, w00, w01, w10, w11) in bini:
        for i in range(2):
            U[i * m + 0][col] = Fraction((u00, u01, u10, u11)[i * 2 + 0])
            U[i * m + 1][col] = Fraction((u00, u01, u10, u11)[i * 2 + 1])
        for j in range(2):
            V[j * p + 0][col] = Fraction((v00, v01, v10, v11)[j * 2 + 0])
            V[j * p + 1][col] = Fraction((v00, v01, v10, v11)[j * 2 + 1])
        for k in range(2):
            W[k * n + 0][col] = Fraction((w00, w01, w10, w11)[k * 2 + 0])
            W[k * n + 1][col] = Fraction((w00, w01, w10, w11)[k * 2 + 1])
        col += 1
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    for i in range(n):
        for j in range(2):
            for k in range(2, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    assert col == rank, (col, rank)
    return U, V, W


def _solve_linear(W_mat, b, n_rows, n_cols):
    """Solve W_mat * x = b over Q. W_mat is n_rows x n_cols, b is length n_rows.
    Returns x (length n_cols) or None if inconsistent."""
    if n_rows == 0:
        return [Fraction(0)] * n_cols
    # Build augmented matrix
    aug = []
    for i in range(n_rows):
        row = W_mat[i][:] + [b[i]]
        aug.append(row)
    # Gaussian elimination
    ncols = n_cols
    rank_r = 0
    for col in range(n_cols):
        piv = -1
        for r in range(rank_r, n_rows):
            if aug[r][col] != 0:
                piv = r
                break
        if piv == -1:
            continue
        aug[rank_r], aug[piv] = aug[piv], aug[rank_r]
        pv = aug[rank_r][col]
        for j in range(col, ncols + 1):
            aug[rank_r][j] /= pv
        for r in range(n_rows):
            if r != rank_r and aug[r][col] != 0:
                f = aug[r][col]
                for j in range(col, ncols + 1):
                    aug[r][j] -= f * aug[rank_r][j]
        rank_r += 1
    # Check consistency
    for r in range(rank_r, n_rows):
        if aug[r][n_cols] != 0:
            return None
    # Extract solution (free vars = 0)
    sol = [Fraction(0)] * n_cols
    for r in range(rank_r):
        pc = -1
        for col in range(n_cols):
            if aug[r][col] != 0:
                pc = col
                break
        if pc != -1:
            sol[pc] = aug[r][n_cols]
    return sol


def _try_rank_r(U_init, V_init, W_init, n, m, p, r, deadline, rng):
    """Try to find a rank-r decomposition starting from initial guesses.
    Uses randomized exact search: fix U and V, solve for W linearly."""
    nU = n * m  # 6
    nV = m * p  # 15
    nW = p * n  # 10
    n_eq = n * m * p  # 30

    # Precompute indices
    idx = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    for attempt in range(2000):
        if time.time() > deadline:
            return None
        # Randomly perturb U and V from initial
        U = [[Fraction(0)] * r for _ in range(nU)]
        V = [[Fraction(0)] * r for _ in range(nV)]
        for i in range(nU):
            for c in range(r):
                # Small rational values
                U[i][c] = Fraction(rng.choice([-2, -1, -1, 0, 0, 1, 1, 2]))
        for j in range(nV):
            for c in range(r):
                V[j][c] = Fraction(rng.choice([-2, -1, -1, 0, 0, 1, 1, 2]))
        # Build linear system for W
        # For each (i,j,k): sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
        # This is linear in W: sum_c (U[i*m+j][c] * V[j*p+k][c]) * W[k*n+i][c] = 1
        W_mat = []
        b = []
        for (i, j, k) in idx:
            row = [Fraction(0)] * nW
            for c in range(r):
                coeff = U[i * m + j][c] * V[j * p + k][c]
                # W index for this term: k*n+i
                row[k * n + i] += coeff
            W_mat.append(row)
            b.append(Fraction(1))
        # Solve
        sol = _solve_linear(W_mat, b, n_eq, nW)
        if sol is None:
            continue
        W = [[Fraction(0)] * r for _ in range(nW)]
        for row_idx in range(nW):
            for c in range(r):
                # sol is indexed by W-row, but we need W[row][c]
                # The linear system treats W as a vector of length nW, where
                # the entry for (k,i) is at position k*n+i.  But W is r-dimensional
                # per row.  This formulation is WRONG: W has r entries per (k,i),
                # not 1.
                pass
        # The above is incorrect.  W has shape (nW, r), so there are nW*r unknowns.
        # The equation for (i,j,k) is:
        #   sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
        # This is linear in the nW*r variables W[k*n+i][c].
        # Let's redo:
        n_unknowns = nW * r
        W_mat = []
        b = []
        for (i, j, k) in idx:
            row = [Fraction(0)] * n_unknowns
            for c in range(r):
                coeff = U[i * m + j][c] * V[j * p + k][c]
                # W variable index: (k*n+i) * r + c
                row[(k * n + i) * r + c] += coeff
            W_mat.append(row)
            b.append(Fraction(1))
        sol = _solve_linear(W_mat, b, n_eq, n_unknowns)
        if sol is None:
            continue
        # Build W
        W = [[Fraction(0)] * r for _ in range(nW)]
        for row_idx in range(nW):
            for c in range(r):
                W[row_idx][c] = sol[row_idx * r + c]
        # Verify
        if _check(U, V, W, n, m, p):
            return U, V, W
    return None


def _try_rank_r_structured(n, m, p, r, deadline, rng):
    """Try rank r using structured patterns: Strassen + random."""
    nU = n * m
    nV = m * p
    nW = p * n
    n_eq = n * m * p

    # Strassen pattern for 2x2x2 (indices i in {0,1}, j in {0,1}, k in {0,1})
    strassen = [
        (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
        (0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0),
        (1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0),
        (0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1),
        (1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0),
        (1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0),
    ]

    for attempt in range(3000):
        if time.time() > deadline:
            return None
        U = [[Fraction(0)] * r for _ in range(nU)]
        V = [[Fraction(0)] * r for _ in range(nV)]
        # Fill first 7 columns with Strassen pattern
        for c in range(min(7, r)):
            (u00, u01, u10, u11, v00, v01, v10, v11, w00, w01, w10, w11) = strassen[c]
            for i in range(2):
                U[i * m + 0][c] = Fraction((u00, u01, u10, u11)[i * 2 + 0])
                U[i * m + 1][c] = Fraction((u00, u01, u10, u11)[i * 2 + 1])
            for j in range(2):
                V[j * p + 0][c] = Fraction((v00, v01, v10, v11)[j * 2 + 0])
                V[j * p + 1][c] = Fraction((v00, v01, v10, v11)[j * 2 + 1])
        # Fill remaining columns randomly
        for c in range(7, r):
            for i in range(nU):
                U[i][c] = Fraction(rng.choice([-1, 0, 1]))
            for j in range(nV):
                V[j][c] = Fraction(rng.choice([-1, 0, 1]))
        # Solve for W
        n_unknowns = nW * r
        W_mat = []
        b = []
        idx = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        for (i, j, k) in idx:
            row = [Fraction(0)] * n_unknowns
            for c in range(r):
                coeff = U[i * m + j][c] * V[j * p + k][c]
                row[(k * n + i) * r + c] += coeff
            W_mat.append(row)
            b.append(Fraction(1))
        sol = _solve_linear(W_mat, b, n_eq, n_unknowns)
        if sol is None:
            continue
        W = [[Fraction(0)] * r for _ in range(nW)]
        for row_idx in range(nW):
            for c in range(r):
                W[row_idx][c] = sol[row_idx * r + c]
        if _check(U, V, W, n, m, p):
            return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin
    rng = random.Random(42)

    # Start with known rank-29
    U29, V29, W29 = _build_rank29(n, m, p)
    best = (U29, V29, W29)
    best_rank = 29

    # Try lower ranks
    for r in range(23, best_rank):
        if time.time() > deadline - 20:
            break
        res = _try_rank_r_structured(n, m, p, r, deadline, rng)
        if res is not None:
            best = res
            best_rank = r
            break  # found minimal in this range

    if not _check(best[0], best[1], best[2], n, m, p):
        best = (U29, V29, W29)

    U, V, W = best
    return {"U": _half(U), "V": _half(V), "W": _half(W)}
