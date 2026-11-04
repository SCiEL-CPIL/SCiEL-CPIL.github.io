"""Strassen-based rank-28 exact tensor decomposition for 2x4 x 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-28 exact decomposition using Strassen's 7-multiplication algorithm
    # applied to 2x2 blocks. Each of the 7 Strassen products M_r is a 2x2 block
    # product, giving 7 * 4 = 28 rank-1 tensors with integer (+/-1) coefficients.
    #
    # Index convention (matches the task's W[k*n+i, r] trace convention):
    #   U[i*m + j, r] = a_{i,j,r}
    #   V[j*p + k, r] = b_{j,k,r}
    #   W[k*n + i, r] = c_{k,i,r}
    # with C[k,i] = sum_r c_{k,i,r} * (sum_{i',j} a_{i',j,r} A[i',j]) * (sum_{j',k'} b_{j',k',r} B[j',k']).
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28

    # Standard Strassen 2x2 formulas. A[i,j], B[j,k] scalars (block indices i,j,k in {0,1}).
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # a_{i,j,r} table (rows i, cols j), b_{j,k,r} (rows j, cols k), c_{k,i,r} (rows k, cols i).
    def term(r):
        if r == 0:  # M1
            a = [[1, 0], [0, 1]]
            b = [[1, 1], [1, 1]]
            c = [[1, 0], [0, 1]]  # c[k,i]: C00 += M1, C11 += M1 -> c[0,0]=1, c[1,1]=1
            return a, b, c
        if r == 1:  # M2
            a = [[0, 0], [1, 1]]  # A10+A11
            b = [[1, 0], [0, 0]]  # B00
            c = [[0, 0], [1, 0]]  # C10 += M2 -> c[1,0]=1
            return a, b, c
        if r == 2:  # M3
            a = [[1, 0], [0, 0]]  # A00
            b = [[0, 1], [0, -1]]  # B01-B11
            c = [[0, 0], [0, 1]]  # C01 += M3 -> c[0,1]=1
            return a, b, c
        if r == 3:  # M4
            a = [[0, 0], [0, 1]]  # A11
            b = [[-1, 0], [1, 0]]  # B10-B00
            c = [[1, 0], [0, 0]]  # C00 += M4 -> c[0,0]=1
            return a, b, c
        if r == 4:  # M5
            a = [[1, 1], [0, 0]]  # A00+A01
            b = [[0, 0], [0, 1]]  # B11
            c = [[-1, 0], [0, 1]]  # C00 -= M5, C01 += M5 -> c[0,0]=-1, c[0,1]=1
            return a, b, c
        if r == 5:  # M6
            a = [[-1, 0], [1, 0]]  # A10-A00
            b = [[1, 1], [0, 0]]  # B00+B01
            c = [[0, 0], [0, 1]]  # C11 += M6 -> c[1,1]=1
            return a, b, c
        if r == 6:  # M7
            a = [[0, 1], [0, -1]]  # A01-A11
            b = [[0, 1], [1, 0]]  # B10+B11
            c = [[1, 0], [0, 0]]  # C00 += M7 -> c[0,0]=1
            return a, b, c
        raise ValueError("bad term index")

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(7):
        a, b, c = term(r)
        base = r * 4
        for i in range(2):
            for j in range(2):
                U[i * m + j][base + i * 2 + j] = a[i][j]
        for j in range(2):
            for k in range(2):
                V[j * p + k][base + j * 2 + k] = b[j][k]
        for k in range(2):
            for i in range(2):
                W[k * n + i][base + k * 2 + i] = c[k][i]

    # Internal verification: reconstruct C = A @ B (with C[k,i] convention) and compare.
    import random
    rng = random.Random(12345)
    A = [[rng.randint(-9, 9) for _ in range(m)] for _ in range(n)]
    B = [[rng.randint(-9, 9) for _ in range(p)] for _ in range(m)]
    C = [[0] * n for _ in range(p)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[k][i] += A[i][j] * B[j][k]
    C_recon = [[0] * n for _ in range(p)]
    for r in range(rank):
        sU = 0
        for i in range(n):
            for j in range(m):
                sU += U[i * m + j][r] * A[i][j]
        sV = 0
        for j in range(m):
            for k in range(p):
                sV += V[j * p + k][r] * B[j][k]
        prod = sU * sV
        for k in range(p):
            for i in range(n):
                C_recon[k][i] += W[k * n + i][r] * prod
    for k in range(p):
        for i in range(n):
            if C_recon[k][i] != C[k][i]:
                # Fallback: should not happen for correct Strassen; but guard against it.
                raise AssertionError("Strassen decomposition verification failed")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
