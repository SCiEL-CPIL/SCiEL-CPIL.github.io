"""Matrix multiplication 6x6x6: Strassen(2x2) x schoolbook(3x3) block decomposition."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-level Strassen decomposition:
    - Partition 6x6 matrices into 3x3 blocks of 2x2 matrices.
    - Apply Strassen (rank 7) at the 2x2 block level.
    - Schoolbook at the 3x3 block level (rank 27).
    - Total rank = 27 * 7 = 189.

    Indexing convention:
    U[i*m + j, r] : coefficient of A[i,j] in rank-r term
    V[j*p + k, r] : coefficient of B[j,k] in rank-r term
    W[k*n + i, r] : coefficient of C[i,k] in rank-r term (official trace convention)

    n=m=p=6.
    """
    n, m, p = DIMENSIONS  # all 6

    # Strassen 2x2 factors: Af[s][il][jl], Bf[s][jl][kl], Cf[s][il][kl]
    # s=0: M1 = (a11+a22)*(b11+b22)
    # s=1: M2 = (a21+a22)*b11
    # s=2: M3 = a11*(b12-b22)
    # s=3: M4 = a22*(b21-b11)
    # s=4: M5 = (a11+a12)*b22
    # s=5: M6 = (a21-a11)*(b11+b12)
    # s=6: M7 = (a12-a22)*(b21+b22)

    Af = [
        [[1, 0], [0, 1]],    # s=0: a11+a22
        [[0, 0], [1, 1]],    # s=1: a21+a22
        [[1, 0], [0, 0]],    # s=2: a11
        [[0, 0], [0, 1]],    # s=3: a22
        [[1, 1], [0, 0]],    # s=4: a11+a12
        [[-1, 0], [1, 0]],   # s=5: a21-a11
        [[0, 1], [0, -1]],   # s=6: a12-a22
    ]
    Bf = [
        [[1, 0], [0, 1]],    # s=0: b11+b22
        [[1, 0], [0, 0]],    # s=1: b11
        [[0, 1], [0, -1]],   # s=2: b12-b22
        [[-1, 0], [1, 0]],   # s=3: b21-b11
        [[0, 0], [0, 1]],    # s=4: b22
        [[1, 1], [0, 0]],    # s=5: b11+b12
        [[0, 0], [1, 1]],    # s=6: b21+b22
    ]
    Cf = [
        [[1, 0], [0, 1]],    # s=0: C11+=M1, C22+=M1
        [[0, 0], [1, -1]],   # s=1: C21+=M2, C22-=M2
        [[0, 1], [0, 1]],    # s=2: C12+=M3, C22+=M3
        [[1, 0], [1, 0]],    # s=3: C11+=M4, C21+=M4
        [[-1, 1], [0, 0]],   # s=4: C11-=M5, C12+=M5
        [[0, 0], [0, 1]],    # s=5: C22+=M6
        [[1, 0], [0, 0]],    # s=6: C11+=M7
    ]

    rank = 27 * 7  # 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189

    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                for s in range(7):
                    # U: A factor
                    for il in range(2):
                        for jl in range(2):
                            i = i_block * 2 + il
                            j = j_block * 2 + jl
                            U[i * m + j][col] += Af[s][il][jl]
                    # V: B factor
                    for jl in range(2):
                        for kl in range(2):
                            j = j_block * 2 + jl
                            k = k_block * 2 + kl
                            V[j * p + k][col] += Bf[s][jl][kl]
                    # W: C factor, W[k*n+i, col]
                    for il in range(2):
                        for kl in range(2):
                            i = i_block * 2 + il
                            k = k_block * 2 + kl
                            W[k * n + i][col] += Cf[s][il][kl]
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
