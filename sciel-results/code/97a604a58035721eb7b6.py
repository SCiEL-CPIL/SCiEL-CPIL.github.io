"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # 20 second safety margin

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = (U_school, V_school, W_school)
    best_rank = rank_school

    # Try to find a lower rank decomposition using randomized search
    # Strategy: pick random U, V with small entries, solve for W, verify
    np.random.seed(42)

    # We know that for 3x5x5, the rank is at most 75 (schoolbook).
    # There are known decompositions with lower rank.
    # For example, if we can find a rank 72 or 71 decomposition.
    # Let's try to search for rank 70 down to 30.
    # The randomized approach:
    # 1. Pick random U (15 x r) and V (25 x r) with small integers.
    # 2. For each (i, k) in [0..2] x [0..4], solve for W[k*n+i, :].
    #    The system is: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j in [0..4].
    #    This is a linear system A @ W_row = b, where A[j, c] = U[i*m+j, c] * V[j*p+k, c].
    #    A is 5 x r. We need an exact solution.
    #    This requires that the rows of A are linearly independent (rank 5) and that b is in the column space.
    #    Since we are solving over reals, we can use lstsq and check the residual.

    for r in range(70, 29, -1):
        if time.time() > deadline:
            break

        found = False
        max_attempts = 200
        for attempt in range(max_attempts):
            # Generate random U and V with small integer entries
            # Using small integers helps with exactness if we could do rational arithmetic,
            # but we are using floats.
            # Let's try entries from {-1, 0, 1} or {-2, -1, 0, 1, 2}.
            U_rand = np.random.randint(-2, 3, size=(n*m, r))
            V_rand = np.random.randint(-2, 3, size=(m*p, r))

            W_rand = np.zeros((p*n, r))
            valid = True

            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U_rand[i*m+j, :] * V_rand[j*p+k, :]
                    b = np.ones(m)

                    # Solve A @ W_row = b
                    try:
                        # Use lstsq
                        W_row, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if the solution is exact
                        if np.allclose(A @ W_row, b, atol=1e-10):
                            W_rand[k*n+i, :] = W_row
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break

                if not valid:
                    break

            if valid:
                # Verify the full decomposition
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rand.tolist()
                if _verify(U_list, V_list, W_list, n, m, p):
                    found = True
                    best_result = (U_list, V_list, W_list)
                    best_rank = r
                    break

        if found:
            # If we found a solution at rank r, we can try lower ranks
            continue

    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
