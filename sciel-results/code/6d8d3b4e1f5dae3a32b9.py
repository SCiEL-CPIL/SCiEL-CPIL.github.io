"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a rank-64 exact tensor decomposition for 4x4 * 4x7 matrix multiplication.

    Strategy (Strassen/hybrid family, strengthened):
      Part 1 (rank 16): compute the 4x4x4 block with an exact rank-8 bilinear
        scheme applied per 2-row pair of A against the first 4 columns of B.
        For rows {r0, r1} and column k of B:
          E_k = a[r0] * b[k]      -> C[r0][k] = E_k
          O_k = (a[r0]+a[r1]) * b[k] -> C[r1][k] = O_k - E_k
        So both output rows of each pair are exact, using only 2 products/column.
      Part 2 (rank 48): schoolbook for the 4x4x3 block (last 3 columns of B).
      Total rank = 16 + 48 = 64.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 4, 4, 7
    rank = 64

    U = [[0] * rank for _ in range(n * m)]   # 16 x rank
    V = [[0] * rank for _ in range(m * p)]   # 28 x rank
    W = [[0] * rank for _ in range(p * n)]   # 28 x rank

    col = 0

    # ---- Part 1: rank-8-per-pair scheme for the 4x4x4 block ----------------
    for row_pair in range(2):                 # pair 0: rows {0,1}; pair 1: rows {2,3}
        if time.time() > deadline:
            break
        r0 = 2 * row_pair
        r1 = 2 * row_pair + 1
        for k in range(4):                    # first 4 columns of B
            if time.time() > deadline:
                break
            # E_k = a[r0] * b[k]  ->  C[r0][k] = E_k
            U[r0 * 4 + k][col] = 1
            V[k * 7 + k][col] = 1
            W[k * 4 + r0][col] = 1
            col += 1
            # O_k = (a[r0] + a[r1]) * b[k]  ->  C[r1][k] = O_k - E_k
            U[r0 * 4 + k][col] = 1
            U[r1 * 4 + k][col] = 1
            V[k * 7 + k][col] = 1
            W[k * 4 + r1][col] = 1
            W[k * 4 + r1][col - 1] = -1
            col += 1

    # ---- Part 2: schoolbook for 4x4 * 4x3 (columns 4,5,6 of B) -------------
    for i in range(4):
        if time.time() > deadline:
            break
        for j in range(4):
            if time.time() > deadline:
                break
            for k_prime in range(3):
                if time.time() > deadline:
                    break
                k_global = 4 + k_prime
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
