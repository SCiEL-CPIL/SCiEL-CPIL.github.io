"""Combined program: schoolbook init + principled rank reduction via linear algebra."""
import time
from fractions import Fraction
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _check_decomposition(n, m, p, U, V, W):
    """Verify exact decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _check_decomposition_float(n, m, p, U, V, W, tol=1e-9):
    """Fast float check."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_contrib_vector(U, V, W, c, n, m, p):
    """Build the flat contribution vector of column c."""
    N = n * m * p
    vec = [Fraction(0)] * N
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec[i * m * p + j * p + k] = (
                    U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                )
    return vec

def _try_reduce_one(U, V, W, n, m, p, rng, max_trials=100):
    """Try to reduce rank by 1 by finding a column whose contribution vector
    is a linear combination of the others, or by a random linear combination
    that zeros out a column's contribution."""
    r = len(U[0])
    if r <= 1:
        return None

    N = n * m * p

    # Build all contribution vectors
    contribs = []
    for c in range(r):
        contribs.append(_build_contrib_vector(U, V, W, c, n, m, p))

    # Strategy 1: For each column, try to express its contribution as a linear
    # combination of the others. If found, we can eliminate it.
    # This is a linear system: contrib_elim = sum_{c != elim} x_c * contrib_c
    # Since N > r-1 typically, it's overdetermined. We use least squares over
    # rationals by checking if the residual is zero.

    # Strategy 2: Random linear combination elimination.
    # Pick two columns a, b. Try to find beta such that the new column
    # (a + beta*b) has zero contribution vector.
    # Contribution of (a + beta*b) is NOT simply contrib_a + beta*contrib_b
    # because the product is trilinear. So we need to compute it directly.

    for trial in range(max_trials):
        if _time_left() < 5:
            return None

        a = rng.randrange(r)
        b = rng.randrange(r)
        if a == b:
            continue

        # Try several rational betas
        betas = [Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2),
                 Fraction(2), Fraction(-2), Fraction(1, 3), Fraction(-1, 3),
                 Fraction(1, 4), Fraction(-1, 4), Fraction(3), Fraction(-3),
                 Fraction(1, 5), Fraction(-1, 5), Fraction(4), Fraction(-4)]
        rng.shuffle(betas)

        for beta in betas:
            if _time_left() < 3:
                return None

            # Form new column a' = a + beta * b
            # U'[i,a'] = U[i,a] + beta * U[i,b]
            # V'[j,a'] = V[j,a] + beta * V[j,b]
            # W'[k,a'] = W[k,a] + beta * W[k,b]
            # Check if contribution of a' is zero
            zero = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U[i * m + j][a] + beta * U[i * m + j][b]
                        v_val = V[j * p + k][a] + beta * V[j * p + k][b]
                        w_val = W[k * n + i][a] + beta * W[k * n + i][b]
                        if u_val * v_val * w_val != Fraction(0):
                            zero = False
                            break
                    if not zero:
                        break
                if not zero:
                    break

            if zero:
                # Column a' has zero contribution, so we can drop it.
                # But we need to verify the remaining columns still sum to 1.
                # Actually, if we replace column a with a' and a' has zero contribution,
                # then the sum is unchanged (since we're adding beta*contrib_b to column a,
                # but the contribution isn't linear, so this doesn't work simply).
                # 
                # Let me reconsider. The issue is that the tensor contribution is
                # trilinear, so we can't simply add columns. Instead, we should:
                # 1. Remove column a entirely
                # 2. Check if the remaining r-1 columns still decompose the tensor
                # 3. If not, try a different approach

                # Actually, the correct approach: we want to find if there exists
                # a set of r-1 columns that decompose the tensor. Let's try
                # removing column a and checking.
                U2 = [row[:a] + row[a+1:] for row in U]
                V2 = [row[:a] + row[a+1:] for row in V]
                W2 = [row[:a] + row[a+1:] for row in W]
                if _check_decomposition(n, m, p, U2, V2, W2):
                    return U2, V2, W2

    # Strategy 3: Try to find a linear relation among all contribution vectors.
    # Build a matrix where each row is a tensor entry and each column is a
    # contribution vector. We want to find a non-trivial null vector.
    # But this is expensive for large r. Instead, try small subsets.

    # Strategy 4: For pairs of columns, check if their contribution vectors
    # are linearly dependent (one is a scalar multiple of the other).
    for a in range(r):
        if _time_left() < 3:
            return None
        for b in range(a + 1, r):
            # Check if contrib_a = alpha * contrib_b for some rational alpha
            # Find a non-zero entry in contrib_b
            found_alpha = None
            for idx in range(N):
                if contribs[b][idx] != Fraction(0):
                    alpha = contribs[a][idx] / contribs[b][idx]
                    found_alpha = alpha
                    break
            if found_alpha is not None:
                # Verify
                if all(contribs[a][idx] == found_alpha * contribs[b][idx] for idx in range(N)):
                    # They are linearly dependent. We can eliminate one.
                    # Remove column b
                    U2 = [row[:b] + row[b+1:] for row in U]
                    V2 = [row[:b] + row[b+1:] for row in V]
                    W2 = [row[:b] + row[b+1:] for row in W]
                    if _check_decomposition(n, m, p, U2, V2, W2):
                        return U2, V2, W2

    return None

def _try_reduce_rank(n, m, p, U, V, W, rng, max_iter=200):
    """Iteratively try to reduce rank."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]

    for iteration in range(max_iter):
        if _time_left() < 5:
            break
        if len(U[0]) <= 1:
            break

        result = _try_reduce_one(U, V, W, n, m, p, rng, max_trials=50)
        if result is not None:
            U, V, W = result
        else:
            break

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Schoolbook initialization
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

    # Verify schoolbook
    assert _check_decomposition(n, m, p, U, V, W), "Schoolbook decomposition failed"

    rng = random.Random(42)

    # Try to reduce rank
    U, V, W = _try_reduce_rank(n, m, p, U, V, W, rng, max_iter=100)

    current_rank = len(U[0])
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank

    # Try random rank-r decompositions for r < current_rank
    for target_rank in range(best_rank - 1, max(15, best_rank - 15), -1):
        if _time_left() < 10:
            break

        for attempt in range(30):
            if _time_left() < 10:
                break

            U_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
            W_rand = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]

            if _check_decomposition_float(n, m, p, U_rand, V_rand, W_rand, tol=1e-6):
                if _check_decomposition(n, m, p, U_rand, V_rand, W_rand):
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    best_rank = target_rank
                    break

    return {"U": best_U, "V": best_V, "W": best_W}
