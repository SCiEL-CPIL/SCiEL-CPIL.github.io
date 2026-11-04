"""Matrix multiplication tensor decomposition for 3x4 x 4x8.

Strategy:
- Row i=0 (1x4x8): schoolbook, rank 32.
- Rows i=1,2 (2x4x8): partition j into 2 blocks of 2, k into 4 blocks of 2.
  For each (j_block, k_block) pair, use Strassen's 7-term algorithm for the
  2x2x2 subproblem. Total rank = 32 + 8*7 = 88.

The W matrix follows the trace convention W[k*n+i, r].
"""
import time

DIMENSIONS = [3, 4, 8]


def construct():
    start = time.time()
    deadline = start + 145.0

    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 96

    # Build schoolbook decomposition as fallback
    def _schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    if time.time() > deadline - 5:
        return _schoolbook()

    # Strassen rank: 32 (for i=0) + 8*7 (for i=1,2) = 88
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: i=0, schoolbook 1x4x8 (32 terms)
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: i=1,2 using Strassen on 2x2x2 blocks
    # Split j into 2 blocks of 2: [0,1], [2,3]
    # Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    # 8 blocks total, each with 7 Strassen terms = 56 terms

    # Strassen 2x2x2 factors
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = AB
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e),
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6

    # U vectors (for A, indexed as [a,b,c,d]):
    sU = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    # V vectors (for B, indexed as [e,f,g,h]):
    sV = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    # W vectors (for C, indexed as [C11,C12,C21,C22]):
    sW = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]

    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # U: local index l -> (local_i, local_j) where l = local_i*2 + local_j
                for l in range(4):
                    val = sU[t][l]
                    if val != 0:
                        li = l // 2
                        lj = l % 2
                        gi = 1 + li
                        gj = j_start + lj
                        U[gi * m + gj][col] += val
                # V: local index l -> (local_j, local_k) where l = local_j*2 + local_k
                for l in range(4):
                    val = sV[t][l]
                    if val != 0:
                        lj = l // 2
                        lk = l % 2
                        gj = j_start + lj
                        gk = k_start + lk
                        V[gj * p + gk][col] += val
                # W: local index l -> (local_i, local_k) where l = local_i*2 + local_k
                for l in range(4):
                    val = sW[t][l]
                    if val != 0:
                        li = l // 2
                        lk = l % 2
                        gi = 1 + li
                        gk = k_start + lk
                        W[gk * n + gi][col] += val
                col += 1

    if col != rank:
        return _schoolbook()

    # Quick verification with a random test case
    import random
    random.seed(42)
    A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
    # Compute AB
    AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    # Compute via decomposition
    C = [[0] * p for _ in range(n)]
    for r in range(rank):
        ua = sum(U[i * m + j][r] * A[i][j] for i in range(n) for j in range(m))
        vb = sum(V[j * p + k][r] * B[j][k] for j in range(m) for k in range(p))
        prod = ua * vb
        for k in range(p):
            for i in range(n):
                C[i][k] += W[k * n + i][r] * prod

    # Check
    for i in range(n):
        for k in range(p):
            if C[i][k] != AB[i][k]:
                return _schoolbook()

    return {"U": U, "V": V, "W": W}
