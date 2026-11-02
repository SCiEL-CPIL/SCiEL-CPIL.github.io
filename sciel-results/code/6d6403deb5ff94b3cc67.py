"""Matrix multiplication 2x2x15 tensor decomposition.

Structural approach: decompose the 2x2x15 tensor into 5 independent 2x2x3
sub-tensors (k grouped as {0,1,2},{3,4,5},...).  Find a small-rank exact
decomposition for 2x2x3 and replicate it 5 times.  A rank-9 block gives total
rank 45; rank-10 gives 50.  Fall back to the known rank-53 Strassen
construction if the block search fails.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140  # safety margin under the 150s kill

    # Baseline: rank-53 Strassen construction (always valid).
    U53, V53, W53 = _build_strassen_53(n, m, p)
    if not _verify_int(U53, V53, W53, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    best_rank = 53
    best_U, best_V, best_W = U53, V53, W53

    # Try to find a small-rank exact decomposition of the 2x2x3 block and
    # replicate it across the 5 triples of k.
    for target_rank in (9, 10):
        if time.time() > deadline - 20:
            break
        block = _find_2x2x3_block(target_rank, deadline)
        if block is None:
            continue
        U10, V10, W10 = block
        U, V, W = _replicate_block(n, m, p, U10, V10, W10)
        if _verify_int(U, V, W, n, m, p):
            rank = len(U[0])
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
            break  # got the best achievable with this block size

    return {"U": best_U, "V": best_V, "W": best_W}


def _build_strassen_53(n, m, p):
    """Rank-53: seven 2x2x2 Strassen blocks (rank 7) + one 2x2x1 schoolbook (rank 4)."""
    n, m, p = 2, 2, 15
    blocks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14,)]
    all_U, all_V, all_W = [], [], []
    for block in blocks:
        if len(block) == 2:
            terms = _strassen_terms(block[0], block[1])
        else:
            terms = _schoolbook_terms(block[0])
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i * m + j, c) for (i, j), c in u_dict.items()]
            v_entries = [(j * p + k_idx, c) for (j, k_idx), c in v_dict.items()]
            w_entries = [(k_idx * n + i, c) for (k_idx, i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    rank = len(all_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val
    return U, V, W


def _strassen_terms(k0, k1):
    """Standard Strassen rank-7 terms for a 2x2x2 block (k indices k0, k1)."""
    return [
        ({(0, 0): 1, (1, 1): 1}, {(0, k0): 1, (1, k1): 1}, {(k0, 0): 1, (k1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, k0): 1}, {(k0, 1): 1, (k1, 1): -1}),
        ({(0, 0): 1}, {(0, k1): 1, (1, k1): -1}, {(k1, 0): 1, (k1, 1): 1}),
        ({(1, 1): 1}, {(1, k0): 1, (0, k0): -1}, {(k0, 0): 1, (k0, 1): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, k1): 1}, {(k0, 0): -1, (k1, 0): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, k0): 1, (0, k1): 1}, {(k1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, k0): 1, (1, k1): 1}, {(k0, 0): 1}),
    ]


def _schoolbook_terms(k):
    """Schoolbook rank-4 terms for a 2x2x1 block."""
    return [
        ({(0, 0): 1}, {(0, k): 1}, {(k, 0): 1}),
        ({(0, 1): 1}, {(1, k): 1}, {(k, 0): 1}),
        ({(1, 0): 1}, {(0, k): 1}, {(k, 1): 1}),
        ({(1, 1): 1}, {(1, k): 1}, {(k, 1): 1}),
    ]


def _build_schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
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
    return U, V, W


def _verify_int(U, V, W, n, m, p):
    """Verify exact integer decomposition: every (i,j,k) entry equals 1."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = 0
                for c in range(rank):
                    val += U[u_row][c] * V[v_row][c] * W[w_row][c]
                if val != 1:
                    return False
    return True


def _replicate_block(n, m, p, U10, V10, W10):
    """Replicate a 2x2x3 block decomposition across the 5 triples of k.

    U10: (4, R) indexed by (i*m+j)
    V10: (6, R) indexed by (j*3 + k_local)
    W10: (6, R) indexed by (k_local*n + i)
    """
    n, m, p = 2, 2, 15
    R = U10.shape[1]
    total_rank = R * 5
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    for t in range(5):
        base = t * R
        k0 = t * 3
        for r in range(R):
            c = base + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j][c] = U10[i * m + j, r]
            for j in range(m):
                for kl in range(3):
                    V[j * p + (k0 + kl)][c] = V10[j * 3 + kl, r]
            for kl in range(3):
                for i in range(n):
                    W[(k0 + kl) * n + i][c] = W10[kl * n + i, r]
    return U, V, W


def _find_2x2x3_block(target_rank, deadline):
    """Find an exact rank-`target_rank` decomposition of the 2x2x3 tensor.

    Strategy: enumerate sparse support patterns for the W factor (each term r
    has a small support in the (k,i) plane), then solve the resulting linear
    system for U and V exactly using rational arithmetic.  The 2x2x3 tensor
    has 12 equations (all equal to 1) and is highly underdetermined, so the
    main task is to land on a solution with small integer / half-integer
    entries.
    """
    n, m, p = 2, 2, 3
    R = target_rank

    # Enumerate candidate support structures for W.  For each term r, choose
    # a small set of (k,i) positions.  We bias toward structures where each
    # (k,i) is covered a bounded number of times so the linear system stays
    # well-conditioned and solutions stay small.
    ki_positions = [(k, i) for k in range(p) for i in range(n)]  # 6 positions

    # Candidate single-term W supports: each is a dict {(k,i): coeff}.
    # We allow coefficient in {1, -1} and support size 1 or 2.
    single_supports = []
    for pos in ki_positions:
        single_supports.append({pos: 1})
        single_supports.append({pos: -1})
    for a, b in product(ki_positions, repeat=2):
        if a == b:
            continue
        single_supports.append({a: 1, b: 1})
        single_supports.append({a: 1, b: -1})
        single_supports.append({a: -1, b: 1})
        single_supports.append({a: -1, b: -1})

    # We'll try to assign, for each of the R terms, one of these supports,
    # then solve.  Full enumeration is too large, so we use a randomized
    # greedy assignment with a time budget, plus a few deterministic
    # structured patterns.
    import random
    rng = random.Random(12345)

    def solve_with_supports(W_supports):
        """Given a list of R W-support dicts, solve for U and V exactly.

        Returns (U, V, W) as Fraction matrices or None.
        """
        # Build the linear system.  Unknowns: U[i,j,r] (4*R) and V[j,k,r] (6*R).
        # Equation for (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
        # This is bilinear in (U,V).  To make it linear, fix one side.
        # We fix V to a simple pattern and solve for U, then check; if that
        # fails, fix U and solve for V.  Because the system is underdetermined
        # and bilinear, we instead use the following exact trick:
        #
        # For each term r, let w_r = W-support dict.  Define for each (i,j,k):
        #   eq(i,j,k): sum_r U[i,j,r]*V[j,k,r]*w_r[k,i] = 1.
        #
        # This is linear in U if V is fixed, and linear in V if U is fixed.
        # We use an alternating exact solve: start with V = identity-like,
        # solve U, then solve V, iterate a few times, then check exactness.
        #
        # Because entries can be fractional, we work in Fraction.

        # Initialize V: for each (j,k), set V[j,k,r] to a small pattern.
        # Use V[j,k,r] = 1 if (j,k) is the "primary" position of term r else 0,
        # to keep it sparse.  Then solve U exactly (linear, underdetermined ->
        # pick the minimum-norm-ish via a simple basis).  Then solve V exactly.
        #
        # Simpler robust approach: set up the full bilinear system and use the
        # fact that with underdetermination we can set most variables to 0 and
        # solve the remaining small linear system.  We do this by choosing, for
        # each equation (i,j,k), a "witness" term r(i,j,k) that will be the
        # main contributor, and solve the resulting per-term constraints.

        # We take a cleaner path: fix W (given), and fix V to be a sparse
        # "selector" so that each (j,k) is selected by exactly one term.  Then
        # each equation (i,j,k) involves exactly one term r, giving
        # U[i,j,r]*V[j,k,r]*W[k,i,r] = 1, i.e. U[i,j,r] = 1/(V[j,k,r]*W[k,i,r]).
        # Choose V[j,k,r] = 1 for the selecting term, then U[i,j,r] = 1/W[k,i,r].
        # But U[i,j,r] must be consistent across the k's for fixed (i,j,r):
        # for a fixed term r and fixed (i,j), U[i,j,r] is a single value, yet
        # it is constrained by each k where V[j,k,r] and W[k,i,r] are nonzero.
        # So we need: for fixed (i,j,r), 1/(V[j,k,r]*W[k,i,r]) is the same for
        # all k in the support.  This is a strong constraint.
        #
        # Given the complexity, we fall back to a direct exact linear solve
        # using the standard "fix V, solve U; fix U, solve V" iteration with
        # Fraction, and check for an exact fixed point.

        # Build W as a Fraction matrix (6 x R).
        Wm = [[Fraction(0)] * R for _ in range(p * n)]
        for r in range(R):
            for (k, i), c in W_supports[r].items():
                Wm[k * n + i][r] = Fraction(c)

        # Initialize V (6 x R) with a sparse pattern: V[j*3+k, r] = 1 if r == (j*3+k) mod R else 0.
        Vm = [[Fraction(0)] * R for _ in range(m * p)]
        for r in range(R):
            idx = r % (m * p)
            Vm[idx][r] = Fraction(1)

        Um = [[Fraction(0)] * R for _ in range(n * m)]

        for _ in range(6):
            # Solve for U given V, W.  For each (i,j): unknowns U[i,j,r] (R of them).
            # Equations: for each k: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.  (3 equations, R unknowns.)
            # Underdetermined; pick the solution with U[i,j,r]=0 for r not in a chosen basis.
            # We choose the basis as the first '3' terms that have nonzero V[j,k,r]*W[k,i,r]
            # collectively spanning.  Simpler: use a least-squares-free exact method by
            # selecting a 3x3 invertible submatrix.
            solved_U = True
            for i in range(n):
                for j in range(m):
                    # Build 3 x R matrix A where A[k, r] = V[j*3+k, r] * W[k*n+i, r].
                    A = [[Fraction(0)] * R for _ in range(p)]
                    for k in range(p):
                        for r in range(R):
                            A[k][r] = Vm[j * p + k][r] * Wm[k * n + i][r]
                    # Need to solve A * u = ones, u in F^R.  Underdetermined.
                    # Pick a set of up to 3 columns that form an invertible 3x3.
                    cols = _pick_invertible_columns(A, p)
                    if cols is None:
                        solved_U = False
                        break
                    u = _solve_small(A, cols, p)
                    if u is None:
                        solved_U = False
                        break
                    for r in range(R):
                        Um[i * m + j][r] = Fraction(0)
                    for r, val in zip(cols, u):
                        Um[i * m + j][r] = val
            if not solved_U:
                break

            # Solve for V given U, W.  For each (j,k): unknowns V[j,k,r] (R).
            # Equations: for each i: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.  (2 equations, R unknowns.)
            solved_V = True
            for j in range(m):
                for k in range(p):
                    A = [[Fraction(0)] * R for _ in range(n)]
                    for i in range(n):
                        for r in range(R):
                            A[i][r] = Um[i * m + j][r] * Wm[k * n + i][r]
                    cols = _pick_invertible_columns(A, n)
                    if cols is None:
                        solved_V = False
                        break
                    v = _solve_small(A, cols, n)
                    if v is None:
                        solved_V = False
                        break
                    for r in range(R):
                        Vm[j * p + k][r] = Fraction(0)
                    for r, val in zip(cols, v):
                        Vm[j * p + k][r] = val
            if not solved_V:
                break

        # Check exactness.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(R):
                        s += Um[i * m + j][r] * Vm[j * p + k][r] * Wm[k * n + i][r]
                    if s != Fraction(1):
                        return None
        return Um, Vm, Wm

    def _pick_invertible_columns(A, rows):
        """Pick up to `rows` columns of A (rows x R) forming an invertible square submatrix."""
        R = len(A[0])
        # Try all combinations of `rows` columns (small R).
        from itertools import combinations
        for combo in combinations(range(R), rows):
            M = [[A[i][c] for c in combo] for i in range(rows)]
            if _det(M) != 0:
                return list(combo)
        return None

    def _det(M):
        n = len(M)
        if n == 0:
            return Fraction(1)
        if n == 1:
            return M[0][0]
        if n == 2:
            return M[0][0] * M[1][1] - M[0][1] * M[1][0]
        # Gaussian elimination for small n.
        M = [row[:] for row in M]
        sign = Fraction(1)
        for col in range(n):
            pivot = None
            for r in range(col, n):
                if M[r][col] != 0:
                    pivot = r
                    break
            if pivot is None:
                return Fraction(0)
            if pivot != col:
                M[col], M[pivot] = M[pivot], M[col]
                sign = -sign
            pv = M[col][col]
            for r in range(col + 1, n):
                factor = M[r][col] / pv
                for c in range(col, n):
                    M[r][c] -= factor * M[col][c]
        d = sign
        for r in range(n):
            d *= M[r][r]
        return d

    def _solve_small(A, cols, rows):
        """Solve A[:, cols] * x = ones using the columns in `cols` (rows x rows invertible)."""
        M = [[A[i][c] for c in cols] for i in range(rows)]
        b = [Fraction(1) for _ in range(rows)]
        # Gaussian elimination.
        n = rows
        M2 = [row[:] for row in M]
        b2 = b[:]
        for col in range(n):
            pivot = None
            for r in range(col, n):
                if M2[r][col] != 0:
                    pivot = r
                    break
            if pivot is None:
                return None
            M2[col], M2[pivot] = M2[pivot], M2[col]
            b2[col], b2[pivot] = b2[pivot], b2[col]
            pv = M2[col][col]
            for r in range(n):
                if r == col:
                    continue
                factor = M2[r][col] / pv
                for c in range(col, n):
                    M2[r][c] -= factor * M2[col][c]
                b2[r] -= factor * b2[col]
        x = [b2[r] / M2[r][r] for r in range(n)]
        return x

    # Deterministic structured patterns first.
    # Pattern A: each term r covers a distinct (k,i) with coeff 1, cycling.
    structured = []
    for pattern_id in range(6):
        supports = []
        for r in range(R):
            pos = ki_positions[(r + pattern_id) % len(ki_positions)]
            supports.append({pos: 1})
        structured.append(supports)

    candidates = list(structured)

    # Randomized supports within budget.
    max_random = 4000
    attempts = 0
    while attempts < max_random and time.time() < deadline - 25:
        supports = []
        for r in range(R):
            s = rng.choice(single_supports)
            supports.append(s)
        candidates.append(supports)
        attempts += 1

    for supports in candidates:
        if time.time() > deadline - 25:
            break
        res = solve_with_supports(supports)
        if res is None:
            continue
        Um, Vm, Wm = res
        # Convert to numpy float arrays (values are Fraction; check they are
        # integer or half-integer).
        Uf, Vf, Wf = [], [], []
        ok = True
        for mat, out in ((Um, Uf), (Vm, Vf), (Wm, Wf)):
            rows = len(mat)
            cols = len(mat[0]) if rows else 0
            arr = np.zeros((rows, cols), dtype=float)
            for i in range(rows):
                for r in range(cols):
                    val = mat[i][r]
                    # Must be integer or half-integer.
                    doubled = val * 2
                    if doubled != doubled.numerator / doubled.denominator:
                        ok = False
                    if val.denominator not in (1, 2):
                        ok = False
                    arr[i, r] = float(val)
            out.append(arr)
        if not ok:
            continue
        Uf = np.array(Uf)
        Vf = np.array(Vf)
        Wf = np.array(Wf)
        # Final float verification.
        if _verify_2x2x3(Uf, Vf, Wf):
            return Uf, Vf, Wf

    return None


def _verify_2x2x3(U, V, W, tol=1e-9):
    """Verify 2x2x3 decomposition numerically."""
    n, m, p = 2, 2, 3
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r] for r in range(R))
                if abs(actual - 1.0) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
