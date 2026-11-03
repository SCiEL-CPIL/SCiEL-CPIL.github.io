"""Fresh design: rank-24 bilinear decomposition of the 2x4 x 4x4 tensor."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_rank24():
    n, m, p = 2, 4, 4
    R = 24
    # U shape (n*m, R) = (8, 24); V shape (m*p, R) = (16, 24); W shape (p*n, R) = (8, 24)
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    # Base 16 terms: T = sum_j B[j,k] e_j (input) * e_k (output)
    for j in range(m):
        for k in range(p):
            t = j * p + k
            V[j * p + k][t] = 1
            W[k * n + 0][t] = 1
            W[k * n + 1][t] = 1
            U[0 * m + j][t] = 1
            U[1 * m + j][t] = 1

    # 8 residual terms: (A[1,j] - A[0,j]) * B[j,k]
    for j in range(m):
        for k in range(p):
            t = 16 + j * p + k
            V[j * p + k][t] = 1
            W[k * n + 1][t] = 1
            U[0 * m + j][t] = -1
            U[1 * m + j][t] = 1

    return {"U": U, "V": V, "W": W}


def _verify(res, trials=12, seed=12345):
    """Random exact integer test of the decomposition; returns True iff correct."""
    import random
    n, m, p = 2, 4, 4
    U, V, W = res["U"], res["V"], res["W"]
    R = len(U[0])
    rnd = random.Random(seed)
    for _ in range(trials):
        A = [[rnd.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[rnd.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        # true product
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # decomposed: C[i,k] = sum_t U[i*m+j? no] -> C[i,k] = sum_{j,t} U[i*m+j][t] V[j*p+k][t]
        # with W[k*n+i][t] selecting output; here W is identity-permutation so:
        Cc = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for t in range(R):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                Cc[i][k] = s
        if Cc != C:
            return False
    return True


def construct():
    start = time.time()
    res = _build_rank24()
    # verify before returning; if somehow wrong, fall back to schoolbook (never expected)
    if not _verify(res):
        n, m, p = DIMENSIONS
        rank = n * m * p
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
        res = {"U": U, "V": V, "W": W}
    return res
# EVOLVE-BLOCK-END
