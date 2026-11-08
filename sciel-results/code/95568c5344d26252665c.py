"""Matrix multiplication 4,4,7 — rank-97 hybrid Strassen(4x4x4) + Schoolbook(4x4x3)."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0

def _deadline_ok():
    return (time.time() - _START) < (_DEADLINE - _MARGIN)

def _strassen_2x2x2():
    # U rows: a00=0, a01=1, a10=2, a11=3
    # V rows: b00=0, b01=1, b10=2, b11=3
    # W rows: c00=0, c01=1, c10=2, c11=3
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0, -1],
    ]
    V = [
        [1, 1,  0, -1, 0, 1,  0],
        [0, 0,  1,  0, 0, 0,  1],
        [0, 0,  0,  1, 0, 1,  0],
        [1, 0, -1,  0, 1, 0,  1],
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0,  1, 0, 0],
        [0, 1, 0, 1,  0, 0, 0],
        [1,-1, 1, 0,  0, 1, 0],
    ]
    return U, V, W

def _tensor_product(U1, V1, W1, U2, V2, W2):
    a1, r1 = len(U1), len(U1[0])
    b1 = len(V1)
    c1 = len(W1)
    a2, r2 = len(U2), len(U2[0])
    b2 = len(V2)
    c2 = len(W2)
    R = r1 * r2
    U = [[0]*R for _ in range(a1*a2)]
    V = [[0]*R for _ in range(b1*b2)]
    W = [[0]*R for _ in range(c1*c2)]
    for r in range(r1):
        for s in range(r2):
            col = r*r2 + s
            for i in range(a1):
                for i2 in range(a2):
                    U[i*a2 + i2][col] = U1[i][r] * U2[i2][s]
            for j in range(b1):
                for j2 in range(b2):
                    V[j*b2 + j2][col] = V1[j][r] * V2[j2][s]
            for k in range(c2):
                for k2 in range(c1):
                    W[k*c1 + k2][col] = W1[k2][r] * W2[k][s]
    return U, V, W

def _schoolbook_4x4x3():
    n, m, p = 4, 4, 3
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = 4, 4, 7

    if not _deadline_ok():
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    U_s, V_s, W_s = _strassen_2x2x2()
    U_444, V_444, W_444 = _tensor_product(U_s, V_s, W_s, U_s, V_s, W_s)

    if not _deadline_ok():
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    U_sb, V_sb, W_sb = _schoolbook_4x4x3()

    r1 = 49
    r2 = 48
    total_rank = r1 + r2

    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]

    for i in range(n*m):
        for r in range(r1):
            U[i][r] = U_444[i][r]
    for j in range(m):
        for k in range(4):
            for r in range(r1):
                V[j*p + k][r] = V_444[j*4 + k][r]
    for k in range(4):
        for i in range(n):
            for r in range(r1):
                W[k*n + i][r] = W_444[k*4 + i][r]

    off = r1
    for i in range(n*m):
        for r in range(r2):
            U[i][off + r] = U_sb[i][r]
    for j in range(m):
        for k in range(3):
            for r in range(r2):
                V[j*p + (k+4)][off + r] = V_sb[j*3 + k][r]
    for k in range(3):
        for i in range(n):
            for r in range(r2):
                W[(k+4)*n + i][off + r] = W_sb[k*4 + i][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
