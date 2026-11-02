"""Matrix multiplication 3,4,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    schoolbook_rank = n * m * p  # 72

    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = schoolbook_rank

    # The multiplication tensor T has entries T[i,j,k] = 1 for all (i,j,k)
    # A rank-r decomposition means:
    #   sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    #
    # Strategy: Use structured random search with small half-integer values.
    # Key insight: we can try rank values from a reasonable lower bound upward.
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Also: rank >= n*m*p / max(n,m,p) = 72/6 = 12
    # So lower bound is 24.
    #
    # We'll use a structured approach:
    # 1. Try known constructions / patterns
    # 2. Random search with half-integer entries
    # 3. Local refinement of near-miss solutions

    def check_decomposition(U, V, W, r):
        """Check if U, V, W form a valid rank-r decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def check_decomposition_fast(U, V, W, r):
        """Faster check using numpy."""
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Half-integer values to try
    half_ints = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    # Try structured patterns first
    # Pattern: use the fact that for small matrices, sometimes rank can be reduced
    # by sharing computation across indices.

    # Try ranks from a reasonable range
    # The theoretical lower bound is 24, but known results suggest
    # rank(3x4x6) might be around 30-40.
    # We'll try from 24 upward, spending more time on lower ranks.

    random.seed(42)

    for rank in range(24, min(schoolbook_rank, 50)):
        if time.time() > deadline:
            break

        # Number of attempts based on remaining time
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        # More attempts for lower ranks (harder to find)
        if rank <= 30:
            max_attempts = int(remaining * 5)
        elif rank <= 40:
            max_attempts = int(remaining * 3)
        else:
            max_attempts = int(remaining * 2)

        max_attempts = max(max_attempts, 10)

        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Random initialization with half-integer values
            # Bias toward small values
            U = [[random.choice(half_ints) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(half_ints) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(half_ints) for _ in range(rank)] for _ in range(p * n)]

            # Quick check
            if check_decomposition(U, V, W, rank):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break

        if found:
            # Try to go lower
            continue
        else:
            # If we couldn't find at this rank, try a few more with different strategy
            # Use a more structured initialization
            for attempt in range(5):
                if time.time() > deadline:
                    break
                # Structured: mostly zeros with a few non-zeros
                U = [[0.0] * rank for _ in range(n * m)]
                V = [[0.0] * rank for _ in range(m * p)]
                W = [[0.0] * rank for _ in range(p * n)]

                # Place non-zero values strategically
                for c in range(rank):
                    # Each term contributes to some subset of (i,j,k)
                    ii = random.randint(0, n - 1)
                    jj = random.randint(0, m - 1)
                    kk = random.randint(0, p - 1)
                    val = random.choice([0.5, 1.0, 1.5, 2.0])
                    U[ii * m + jj][c] = val
                    V[jj * p + kk][c] = 1.0 / val if val != 0 else 0.0
                    W[kk * n + ii][c] = 1.0

                if check_decomposition(U, V, W, rank):
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
                    found = True
                    break

        if found:
            continue

        # If rank is getting high and we've spent a lot of time, stop
        if rank >= 40 and time.time() > start_time + 80:
            break

    # One more pass: try to refine by starting from schoolbook and removing columns
    # This is a greedy rank reduction
    if time.time() < deadline - 10:
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

        # Try to merge/remove columns
        # For each pair of columns, check if one can be removed
        # by adjusting others (this is complex, so skip for now)

    return best_result
# EVOLVE-BLOCK-END
