import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 2, 9

    # Rank-32 decomposition: 4 Strassen blocks (7 each) + 1 schoolbook block (4)
    # Split columns: (0,1), (2,3), (4,5), (6), (7,8)
    rank = 32
    U = [[0] * rank for _ in range(n * m)]   # (4, 32)
    V = [[0] * rank for _ in range(m * p)]   # (18, 32)
    W = [[0] * rank for _ in range(p * n)]   # (18, 32)

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # 1. (a00+a11)(b00+b11)
        U[0][col] = 1; U[3][col] = 1
        V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        # 2. (a10+a11)b00
        U[2][col] = 1; U[3][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
        col += 1
        # 3. a00(b01-b11)
        U[0][col] = 1
        V[0 * p + k1][col] = 1; V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        # 4. a11(b10-b00)
        U[3][col] = 1
        V[1 * p + k0][col] = 1; V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
        col += 1
        # 5. (a00+a01)b11
        U[0][col] = 1; U[1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
        col += 1
        # 6. (a10-a00)(b00+b01)
        U[2][col] = 1; U[0][col] = -1
        V[0 * p + k0][col] = 1; V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        # 7. (a01-a11)(b10+b11)
        U[1][col] = 1; U[3][col] = -1
        V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1; V[0 * p + k][col] = 1; W[k * n + 0][col] = 1
        col += 1
        U[1][col] = 1; V[1 * p + k][col] = 1; W[k * n + 0][col] = 1
        col += 1
        U[2][col] = 1; V[0 * p + k][col] = 1; W[k * n + 1][col] = 1
        col += 1
        U[3][col] = 1; V[1 * p + k][col] = 1; W[k * n + 1][col] = 1
        col += 1

    # Build the decomposition
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_trivial(6)
    add_strassen(7, 8)

    # Verify: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank-36
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col_fb = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col_fb] = 1
                V_fb[j * p + k][col_fb] = 1
                W_fb[k * n + i][col_fb] = 1
                col_fb += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
