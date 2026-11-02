"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 150.0  # wall-clock limit
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
        # For each output entry (i,k), the sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # should equal delta_{i,j}... wait, let me think about the tensor structure.
        #
        # The matrix multiplication tensor T[i,j,k] = 1 if and only if the product C[i,k] = sum_j A[i,j]*B[j,k]
        # has a term involving A[i,j]*B[j,k]*C[k,i] (using trace convention W[k*n+i]).
        #
        # Actually, the decomposition says:
        # T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # where T is the 3D tensor with T[i,j,k] = 1 (for all i,j,k in the multiplication).
        #
        # The standard matrix multiplication tensor has entries:
        # T[i,j,k] = 1 for all valid (i,j,k)
        # and the decomposition is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        #
        # Wait, that's not right either. Let me reconsider.
        #
        # The tensor for matrix multiplication C = A * B where A is n×m and B is m×p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        #
        # The multiplication tensor T has indices (i, j, k) where:
        # - i indexes rows of A (0..n-1)
        # - j indexes columns of A / rows of B (0..m-1)
        # - k indexes columns of B (0..p-1)
        #
        # T[i,j,k] = 1 for all valid (i,j,k)
        #
        # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        #
        # So we need: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

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
    # Known lower bounds and constructions:
    # - Strassen's algorithm gives rank 7 for 2x2*2x2
    # - For larger matrices, various constructions exist
    #
    # Let's try a systematic search for low-rank decompositions using integer/half-integer entries.
    # We'll use a randomized approach with local search.

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
    # Theoretical lower bound for 3x3 * 3x8:
    # The tensor rank is at least max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But actual lower bounds are higher.
    # Let's try a few candidate ranks.

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
