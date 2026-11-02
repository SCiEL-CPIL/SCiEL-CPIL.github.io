"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # wall-clock limit with safety margin
    SAFETY_MARGIN = 5.0  # seconds to spare

    n, m, p = DIMENSIONS

    # Schoolbook decomposition as fallback
    def schoolbook():
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

    def check_valid(U, V, W):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Try to find a lower-rank decomposition using known results
    # For 3x3 * 3x8, the schoolbook rank is 3*3*8 = 72.
    # Let's try a systematic search for low-rank decompositions using integer/half-integer entries.

    import random
    import numpy as np

    def random_decomposition(rank, max_val=3):
        """Generate a random decomposition with integer entries."""
        U = [[random.randint(-max_val, max_val) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.randint(-max_val, max_val) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.randint(-max_val, max_val) for _ in range(rank)] for _ in range(p * n)]
        return U, V, W

    def error(U, V, W):
        """Compute the error of the decomposition."""
        rank = len(U[0]) if U else 0
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    max_err = max(max_err, abs(s - 1))
        return max_err

    def local_search(U, V, W, iterations=1000):
        """Improve a decomposition via local search."""
        rank = len(U[0]) if U else 0
        best_err = error(U, V, W)
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]

        for it in range(iterations):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            # Pick a random position to modify
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, n * m - 1)
                col = random.randint(0, rank - 1)
                old_val = U[row][col]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                if abs(new_val) > 5:
                    continue
                U[row][col] = new_val
                err = error(U, V, W)
                if err <= best_err:
                    best_err = err
                else:
                    U[row][col] = old_val
            elif mat == 'V':
                row = random.randint(0, m * p - 1)
                col = random.randint(0, rank - 1)
                old_val = V[row][col]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                if abs(new_val) > 5:
                    continue
                V[row][col] = new_val
                err = error(U, V, W)
                if err <= best_err:
                    best_err = err
                else:
                    V[row][col] = old_val
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, rank - 1)
                old_val = W[row][col]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                if abs(new_val) > 5:
                    continue
                W[row][col] = new_val
                err = error(U, V, W)
                if err <= best_err:
                    best_err = err
                else:
                    W[row][col] = old_val

            if best_err == 0:
                break

        return U, V, W, best_err

    # Try ranks from low to high
    best_result = None
    best_rank = n * m * p  # schoolbook

    # Try to find a better decomposition
    for rank in range(24, best_rank + 1):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break

        # Try a few random initializations
        for trial in range(10):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break

            U, V, W = random_decomposition(rank, max_val=2)
            U, V, W, err = local_search(U, V, W, iterations=500)

            if err == 0:
                if check_valid(U, V, W):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
                    break
        else:
            # Continue to next rank if no solution found
            continue
        break

    if best_result is not None:
        return best_result

    # Return schoolbook as fallback
    return schoolbook()
# EVOLVE-BLOCK-END
