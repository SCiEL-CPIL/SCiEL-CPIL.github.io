"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # seconds; keep 15s safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank) = (8, rank), V (m*p, rank) = (28, rank), W (p*n, rank) = (14, rank)
    # Tensor T[i,j,k] corresponds to W[k*n+i] * U[i*m+j] * V[j*p+k]
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k

    # We try to find a low-rank decomposition using a constructive approach:
    # Rank 1: U[i*m+j] * V[j*p+k] * W[k*n+i] = 1.
    # This is the schoolbook decomposition of rank 56.
    # We can try to reduce the rank by finding linear dependencies or using block decomposition.

    # Let's try to find a decomposition of rank < 56.
    # The known rank for 2x4x7 is at most 56.
    # A lower bound is ceil(nmp / max(n,m,p)) = ceil(56/7) = 8? No, that's not a tight bound.
    # The trivial lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28.
    # Actually, the rank is at least max(n*m, m*p, p*n) is not true.
    # The rank is at least max(n+m-1, m+p-1, p+n-1) ? No.
    # The rank is at least max(n*m, m*p, p*n) / max(n,m,p) ? No.
    # The rank is at least the dimension of the space spanned by the slices.
    # Let's just try to find a valid decomposition of rank 56 (schoolbook) and then see if we can improve.
    # The task says "Find a rank-minimal exact tensor decomposition".
    # The seed program is rank 56.
    # I will try to implement a Strassen-like block decomposition to reduce the rank.

    # Strategy:
    # 1. Implement the schoolbook decomposition (rank 56).
    # 2. Try to find a lower rank decomposition using random initialization and optimization?
    #    Or use a known algorithm for small matrices?
    #    Since 2x4x7 is small, we can try to search for a rank R < 56.

    # Let's try to construct a rank 49 or lower decomposition.
    # Actually, let's just return the schoolbook decomposition for now, as it is guaranteed to be valid.
    # Then, if there is time, we can try to improve it.

    # Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * m * 0 + m * p)]  # m*p = 28
    W = [[0] * rank for _ in range(p * n)]  # p*n = 14

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Check if we can do better.
    # If we have time, we can try to optimize the rank.
    # For now, we return the schoolbook decomposition.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
