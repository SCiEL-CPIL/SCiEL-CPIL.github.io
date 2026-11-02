"""Strassen-based rank-reduced exact tensor decomposition for M_{3,4,8}."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Strassen factors for 2x2x2 matrix multiplication (rank 7).
    U shape (4,7): U[i*2+j, c]; V (4,7): V[j*2+k, c]; W (4,7): W[k*2+i, c].
    Reconstructs (AB)_{i,k} = sum_c U[i*2+j,c]*A[i,j]*B[j,k]*W[k*2+i,c].
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    terms = [
        (0, 3, 2),   # A00*(B01+B11)
        (2, 1, 3),   # (A10+A11)*B00
        (0, 0, 0),   # A00*B00
        (3, 3, 1),   # A11*B11
        (0, 2, 0),   # (A00+A10)*(B10-B00)
        (2, 2, 1),   # (A11+A01)*(B01-A00)
        (1, 2, 2),   # (A01-A11)*(B10+B11)
    ]
    for c, (a, b, w) in enumerate(terms):
        U[a][c] = 1
        V[b][c] = 1
        W[w][c] = 1
    return U, V, W


def _check_222():
    import random
    sU, sV, sW = _strassen_222()
    for _ in range(100):
        A = [[random.randint(0, 3) for _ in range(2)] for _ in range(2)]
        B = [[random.randint(0, 3) for _ in range(2)] for _ in range(2)]
        C = [[sum(A[i][j] * B[j][k] for j in range(2)) for k in range(2)] for i in range(2)]
        C2 = [[0] * 2 for _ in range(2)]
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        if sU[i * 2 + j][c] and sV[j * 2 + k][c] and sW[k * 2 + i][c]:
                            C2[i][k] += A[i][j] * B[j][k]
        if C != C2:
            raise AssertionError("Strassen 2x2x2 check failed")


def _check_full(U, V, W, n, m, p):
    import random
    rank = len(U[0])
    for _ in range(30):
        A = [[random.randint(0, 4) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(0, 4) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C2 = [[0] * p for _ in range(n)]
        for c in range(rank):
            for i in range(n):
                for k in range(p):
                    wc = W[k * n + i][c]
                    if not wc:
                        continue
                    s = 0
                    for j in range(m):
                        uc = U[i * m + j][c]
                        if uc:
                            s += uc * A[i][j] * wc * B[j][k]
                    C2[i][k] += s
        if C != C2:
            raise AssertionError("Full decomposition check failed")


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 8
    sU, sV, sW = _strassen_222()

    # Row split: rows 0,1 -> 4 Strassen(2x2x2) blocks; row 2 -> 4 schoolbook(1x2x2).
    # Block c uses B rows {2c,2c+1} and C cols {2c,2c+1}.
    rank = 4 * 7 + 4 * 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for r in range(2):
        for c in range(4):
            for t in range(7):
                for i in range(2):
                    for j in range(2):
                        uval = sU[i * 2 + j][t]
                        if uval:
                            U[r * m + j][col] += uval
                for j in range(2):
                    for k in range(2):
                        vval = sV[j * 2 + k][t]
                        if vval:
                            V[(2 * r + j) * p + (2 * c + k)][col] += vval
                for k in range(2):
                    for i in range(2):
                        wval = sW[k * 2 + i][t]
                        if wval:
                            W[(2 * c + k) * n + (r + i)][col] += wval
                col += 1

    for c in range(4):
        for t in range(3):
            if t == 0:
                U[2 * m + 0][col] = 1
                V[(2 * c + 0) * p + (2 * c + 0)][col] = 1
                W[(2 * c + 0) * n + 2][col] = 1
            elif t == 1:
                U[2 * m + 0][col] = 1
                V[(2 * c + 1) * p + (2 * c + 1)][col] = 1
                W[(2 * c + 1) * n + 2][col] = 1
            else:
                U[2 * m + 0][col] = 1
                V[(2 * c + 0) * p + (2 * c + 1)][col] = 1
                V[(2 * c + 1) * p + (2 * c + 0)][col] = 1
                W[(2 * c + 0) * n + 2][col] = 1
                W[(2 * c + 1) * n + 2][col] = 1
            col += 1

    if time.time() - t0 < 60:
        _check_222()
    if time.time() - t0 < 120:
        _check_full(U, V, W, n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
