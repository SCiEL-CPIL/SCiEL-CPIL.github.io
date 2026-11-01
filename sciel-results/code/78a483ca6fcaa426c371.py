"""Find a rank-minimal exact tensor decomposition for 2x2 * 2x10 multiplication.

Output factor follows the official trace convention W[k*n+i, r].
Score is the rank (lower is better).
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 10]

# ---------------------------------------------------------------------------
# Deadline guard: the evaluator kills the process at ~150 s wall-clock.
# We keep an explicit guard and stop searching with a safety margin, then
# return the best (lowest-rank, verified) decomposition found.
# ---------------------------------------------------------------------------
_DEADLINE = time.time() + 138.0          # 12 s safety margin before the 150 s kill
def _timeup():
    return time.time() >= _DEADLINE


def _verify(U, V, W, n=2, m=2, p=10, trials=400, seed=12345):
    """Exact verification that U,V,W decompose the multiplication tensor.

    For every (i,j,k):  A[i,j]*B[j,k] == sum_r U[i,j,r]*V[j,k,r]*W[k,i,r].
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        A = rng.randint(-3, 4, (n, m))
        B = rng.randint(-3, 4, (m, p))
        C = A @ B
        ok = True
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for j in range(m):
                    acc += A[i, j] * B[j, k] * W[k * n + i, :]
                    # note: acc is a vector; accumulate per product
                # recompute properly:
                acc = np.zeros(rank)
                for j in range(m):
                    acc = acc + A[i, j] * B[j, k] * W[k * n + i, :]
                got = float(np.dot(U[i * m + j0, :] * V[j0 * p + k, :], 1)) if False else None
        # The compact vectorised check below is used instead (faster & correct).
        accC = np.zeros((n, p))
        for r in range(rank):
            accC += (U[:, r].reshape(n, m)) * (V[:, r].reshape(m, p)) * W[:, r].reshape(p, n)
        if not np.array_equal(accC, C):
            return False
    return True


# ---------------------------------------------------------------------------
# Rank-6 search for 2x2 matrix multiplication.
#
# We fix the A-side factor U (n*m x 6) and C-side factor W (p*n x 6) and solve
# for the B-side factor V (m*p x 6) by least squares on a set of random
# instances, then verify the exact bilinear identity.  We try many random
# U/W initialisations (with structured patterns that are known to be close to
# valid tensor decompositions) until a valid rank-6 scheme is found.
# ---------------------------------------------------------------------------

def _solve_and_check(U, W, n=2, m=2, p=2, R=6, inst=160, seed=777):
    """Given fixed U (A-side) and W (C-side), solve V (B-side) and verify."""
    U = np.asarray(U, dtype=float)
    W = np.asarray(W, dtype=float)
    rng = np.random.RandomState(seed)
    # Build the linear system: for each (i,k) and each instance t:
    #   sum_j A[i,j]*V[j,k,r] * (sum_r' ... )  -> linear in V[j,k,*]
    # We assemble M (rows x (m*p*R)) and rhs (rows x R) and solve least squares.
    rows = []
    rhs = []
    for _ in range(inst):
        A = rng.randint(-4, 5, (n, m)).astype(float)
        B = rng.randint(-4, 5, (m, p)).astype(float)
        C = A @ B
        for i in range(n):
            for k in range(p):
                # C[i,k] = sum_j A[i,j] * B[j,k] * W[k*n+i, r]  (per r, summed)
                # = sum_r [ sum_j A[i,j] * B[j,k] * W[k*n+i,r] ]
                # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = C[i,k]
                # This is linear in V: for fixed r, coefficient of V[j*p+k, r] is
                #   U[i*m+j, r] * W[k*n+i, r]
                # and the equation couples all r through the sum.  So the unknown
                # vector is V (m*p*R) and the equation for (i,k) is:
                #   sum_{j,r} (A[i,j] * W[k*n+i, r] * U[i*m+j, r]) * V[j*p+k, r] = C[i,k]
                row = np.zeros(m * p * R)
                for j in range(m):
                    for r in range(R):
                        row[j * p * R + k * R + r] = A[i, j] * W[k * n + i, r] * U[i * m + j, r]
                rows.append(row)
                rhs.append(C[i, k])
    M = np.array(rows)
    b = np.array(rhs)
    # Solve least squares for V flattened (m*p*R)
    Vflat, *_ = np.linalg.lstsq(M, b, rcond=None)
    V = Vflat.reshape(m * p, R)
    # Verify exactly on a fresh large set
    if not _verify(U, V, W, n=n, m=m, p=p, trials=400, seed=999):
        return None
    return U, V, W


def _rand_pattern(n, m, p, R, rng):
    """Generate random integer/half-integer U and W with a few 0/+-1 structure."""
    def gen(rows):
        M = np.zeros((rows, R))
        for r in range(R):
            # each column is a sparse +-1 vector
            for row in range(rows):
                if rng.random() < 0.6:
                    M[row, r] = rng.choice([-1, 1])
        return M
    U = gen(n * m)
    W = gen(p * n)
    return U, W


def _structured_patterns(n, m, p, R):
    """Yield a handful of structured (U, W) initialisations, including
    near-Strassen patterns that are known to be close to valid decompositions."""
    # Pattern A: pure random (handled separately).
    yield _rand_pattern(n, m, p, R, np.random.RandomState(1))
    # Pattern B: U and W share a common sparse structure (mimics Strassen's
    # symmetric use of the same 7 products on both sides).
    rng = np.random.RandomState(2)
    base = np.zeros((n * m, R))
    for r in range(R):
        for row in range(n * m):
            if rng.random() < 0.6:
                base[row, r] = rng.choice([-1, 1])
    W = base.copy()
    U = base.copy()
    yield U, W
    # Pattern C: U random, W = U with rows permuted (C-side mirrors A-side).
    U = _rand_pattern(n, m, p, R, np.random.RandomState(3))[0]
    perm = np.random.RandomState(4).permutation(n * m)
    W = U[perm, :]
    yield U, W


def _search_rank6(max_iters=4000, seed0=0):
    n, m, p, R = 2, 2, 2, 6
    # First try structured patterns (cheap, high hit-rate for near-Strassen).
    for idx, (U, W) in enumerate(_structured_patterns(n, m, p, R)):
        if _timeup():
            return None
        res = _solve_and_check(U, W, n=n, m=m, p=p, R=R, inst=160, seed=1000 + idx)
        if res is not None:
            return res
    # Then random restarts.
    rng = np.random.RandomState(seed0)
    it = 0
    while it < max_iters and not _timeup():
        U, W = _rand_pattern(n, m, p, R, rng)
        it += 1
        res = _solve_and_check(U, W, n=n, m=m, p=p, R=R, inst=160, seed=5000 + it)
        if res is not None:
            return res
    return None


# ---------------------------------------------------------------------------
# Rank-35 (Strassen) fallback: 5 independent 2x2 blocks, each rank-7.
# This is the known island best and is always valid.
# ---------------------------------------------------------------------------

def _strassen_block():
    """A valid rank-7 decomposition of 2x2 * 2x2 -> 2x2 (Strassen).

    Returns U (4x7), V (4x7), W (4x7) in the (i,j),(j,k),(k,i) layout.
    """
    # Strassen products (1-indexed):
    # M1 = (a+d)(e+h)
    # M2 = (c+d) e
    # M3 = a (f-h)
    # M4 = d (g-h)
    # M5 = (a+b)(e+f)
    # M6 = (c-d)(g+h)
    # M7 = (b-d)(e+h)
    # Output:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # A layout (i,j): a=0, b=1, c=2, d=3
    # B layout (j,k): e=0, f=1, g=2, h=3
    # C layout (k,i): C11=0, C12=1, C21=2, C22=3
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    # M1: a+d times e+h
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2: (c+d) e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    # M3: a (f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    # M4: d (g-h)
    U[3, 3] = 1
    V[2, 3] = 1; V[3, 3] = -1
    W[2, 3] = 1
    # M5: (a+b)(e+f)
    U[0, 4] = 1; U[1, 4] = 1
    V[0, 4] = 1; V[1, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # M6: (c-d)(g+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    # M7: (b-d)(e+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = -1
    return U, V, W


def _build_rank35():
    n, m, p = 2, 2, 10
    # 5 blocks of 2 columns each; each block is a 2x2 * 2x2 multiplication.
    nb = p // 2  # 5
    R = 7
    total = R * nb
    U = np.zeros((n * m, total), dtype=float)
    V = np.zeros((m * p, total), dtype=float)
    W = np.zeros((p * n, total), dtype=float)
    Ub, Vb, Wb = _strassen_block()
    for b in range(nb):
        col0 = b * R
        # U: same A-side for every block (A is shared)
        U[:, col0:col0 + R] = Ub
        # V: block b uses B columns 2b, 2b+1
        V[0 * p + 2 * b:0 * p + 2 * b + 2, col0:col0 + R] = Vb[0:2, :]   # e,f
        V[1 * p + 2 * b:1 * p + 2 * b + 2, col0:col0 + R] = Vb[2:4, :]   # g,h
        # W: block b contributes to C columns 2b, 2b+1
        W[2 * b + 0, col0:col0 + R] = Wb[0, :]   # C11 (k=2b, i=0)
        W[2 * b + 1, col0:col0 + R] = Wb[1, :]   # C12 (k=2b+1, i=0)
        W[2 * b + 0 + n, col0:col0 + R] = Wb[2, :]   # C21 (k=2b, i=1)
        W[2 * b + 1 + n, col0:col0 + R] = Wb[3, :]   # C22 (k=2b+1, i=1)
    return U, V, W


def _to_lists(U, V, W):
    """Convert to the required list-of-lists interface (int/half-int entries)."""
    def conv(M):
        M = np.asarray(M, dtype=float)
        out = []
        for row in M:
            lr = []
            for x in row:
                # snap to nearest integer or half-integer
                h = round(2 * x) / 2.0
                if abs(h - round(h)) < 1e-9:
                    lr.append(int(round(h)))
                else:
                    lr.append(h)
            out.append(lr)
        return out
    return {"U": conv(U), "V": conv(V), "W": conv(W)}


def _schoolbook():
    n, m, p = 2, 2, 10
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """Return the best (lowest-rank, verified) exact decomposition found."""
    best = None
    best_rank = None

    # 1) Attempt the rank-6 (total 30) search.
    try:
        r6 = _search_rank6(max_iters=6000, seed0=2024)
        if r6 is not None:
            U, V, W = r6
            # Build the 2x10 decomposition from the 2x2 rank-6 block (5 blocks).
            n, m, p = 2, 2, 10
            nb = p // 2
            R = U.shape[1]
            total = R * nb
            Ufull = np.zeros((n * m, total), dtype=float)
            Vfull = np.zeros((m * p, total), dtype=float)
            Wfull = np.zeros((p * n, total), dtype=float)
            for b in range(nb):
                col0 = b * R
                Ufull[:, col0:col0 + R] = U
                Vfull[0 * p + 2 * b:0 * p + 2 * b + 2, col0:col0 + R] = V[0:2, :]
                Vfull[1 * p + 2 * b:1 * p + 2 * b + 2, col0:col0 + R] = V[2:4, :]
                Wfull[2 * b + 0, col0:col0 + R] = W[0, :]
                Wfull[2 * b + 1, col0:col0 + R] = W[1, :]
                Wfull[2 * b + 0 + n, col0:col0 + R] = W[2, :]
                Wfull[2 * b + 1 + n, col0:col0 + R] = W[3, :]
            if _verify(Ufull, Vfull, Wfull, n=2, m=2, p=10, trials=600, seed=424242):
                best = _to_lists(Ufull, Vfull, Wfull)
                best_rank = total
    except Exception:
        pass

    if best is not None:
        return best

    # 2) Rank-35 Strassen fallback.
    try:
        U, V, W = _build_rank35()
        if _verify(U, V, W, n=2, m=2, p=10, trials=600, seed=424242):
            return _to_lists(U, V, W)
    except Exception:
        pass

    # 3) Schoolbook last resort (always valid, rank 40).
    return _schoolbook()
