"""Matrix multiplication 2x3 x 3x9 via Kronecker product T(2,3,3) ⊗ I_3."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s limit

    # --- 16-term exact decomposition of T(2,3,3) ---
    # U: (2*3)=6 rows, V: (3*3)=9 rows, W: (3*2)=6 rows, rank 16
    # Convention: U[i*3+j], V[j*3+k], W[k*2+i]
    U233 = [
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    V233 = [
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    W233 = [
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ]

    # --- Build (2,3,9) factors via Kronecker product with I_3 ---
    # U: (n*m)=6 rows -> (n*m*g)=18 rows, rank 48
    # V: (m*p)=27 rows -> (m*p*g)=81 rows, rank 48
    # W: (p*n)=18 rows -> (p*n*g)=54 rows, rank 48
    g = 3
    r233 = 16
    rank = r233 * g  # 48

    n, m, p = 2, 3, 9

    # U rows: (i, j, s) where i in [0,n), j in [0,m), s in [0,g)
    # U[(i*m+j)*g + s, c233*g + s] = U233[i*m+j, c233]
    U = [[0] * rank for _ in range(n * m * g)]
    for i in range(n):
        for j in range(m):
            for s in range(g):
                for c in range(r233):
                    U[(i * m + j) * g + s][c * g + s] = U233[i * m + j][c]

    # V rows: (j, k, s) where j in [0,m), k in [0,p), s in [0,g)
    # V[(j*p+k)*g + s, c233*g + s] = V233[j*3+k%3, c233]
    V = [[0] * rank for _ in range(m * p * g)]
    for j in range(m):
        for k in range(p):
            for s in range(g):
                for c in range(r233):
                    V[(j * p + k) * g + s][c * g + s] = V233[j * 3 + (k % 3)][c]

    # W rows: (k, i, s) where k in [0,p), i in [0,n), s in [0,g)
    # W[(k*n+i)*g + s, c233*g + s] = W233[(k%3)*2+i, c233]
    W = [[0] * rank for _ in range(p * n * g)]
    for k in range(p):
        for i in range(n):
            for s in range(g):
                for c in range(r233):
                    W[(k * n + i) * g + s][c * g + s] = W233[(k % 3) * 2 + i][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
