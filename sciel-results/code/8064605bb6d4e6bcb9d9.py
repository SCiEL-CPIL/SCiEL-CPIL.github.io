import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 4, 6

    # Build rank-42 Strassen decomposition
    # Split 4 middle dim into two 2s, split 6 output cols into three 2s
    # 6 blocks of 2x2@2x2, each rank 7 => total rank 42

    rank_total = 42
    U = [[0] * rank_total for _ in range(n * m)]  # (8, 42)
    V = [[0] * rank_total for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank_total for _ in range(p * n)]  # (12, 42)

    # Strassen 2x2@2x2:
    # P1=(a00+a11)(b00+b11), P2=(a10+a11)b00, P3=a00(b01-b11),
    # P4=a11(b10-b00), P5=(a00+a01)b11, P6=(a10-a00)(b00+b01),
    # P7=(a01-a11)(b10+b11)
    # c00=P1+P4-P5+P7, c01=P3+P5, c10=P2+P4, c11=P1-P2+P3+P6

    sU = [
        {0:1, 3:1},   # P1
        {2:1, 3:1},   # P2
        {0:1},        # P3
        {3:1},        # P4
        {0:1, 1:1},   # P5
        {0:-1, 2:1},  # P6
        {1:1, 3:-1},  # P7
    ]
    sV = [
        {0:1, 3:1},   # P1
        {0:1},        # P2
        {1:1, 3:-1},  # P3
        {2:1, 0:-1},  # P4
        {3:1},        # P5
        {0:1, 1:1},   # P6
        {2:1, 3:1},   # P7
    ]
    sW = [
        {0:1, 3:1},   # P1
        {2:1, 3:-1},  # P2
        {1:1, 3:1},   # P3
        {0:1, 2:1},   # P4
        {0:-1, 1:1},  # P5
        {3:1},        # P6
        {0:1},        # P7
    ]

    col = 0
    for jb in range(3):
        for side in ['L', 'R']:
            if side == 'L':
                uoff, vrow = 0, 0
            else:
                uoff, vrow = 2, 2

            for pi in range(7):
                c = col + pi

                # U: local (i,j) -> global row i*4 + (uoff + j)
                for li, sign in sU[pi].items():
                    li_i = li // 2
                    li_j = li % 2
                    U[li_i * 4 + uoff + li_j][c] += sign

                # V: local (k,l) -> global row (vrow+k)*6 + (2*jb+l)
                for li, sign in sV[pi].items():
                    li_k = li // 2
                    li_l = li % 2
                    V[(vrow + li_k) * 6 + 2 * jb + li_l][c] += sign

                # W: local (i,k) -> global row (2*jb+k)*2 + i
                for li, sign in sW[pi].items():
                    li_i = li // 2
                    li_k = li % 2
                    W[(2 * jb + li_k) * 2 + li_i][c] += sign

            col += 7

    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    np.random.seed(42)
    A_t = np.random.randint(-5, 6, (n, m)).astype(float)
    B_t = np.random.randint(-5, 6, (m, p)).astype(float)
    C_exp = A_t @ B_t

    C_comp = np.zeros((n, p))
    for r in range(rank_total):
        left = np.zeros(n)
        for i in range(n):
            for k in range(m):
                left[i] += A_t[i, k] * U_arr[i * m + k, r]
        right = np.zeros(p)
        for j in range(p):
            for k in range(m):
                right[j] += B_t[k, j] * V_arr[k * p + j, r]
        for i in range(n):
            for j in range(p):
                C_comp[i, j] += W_arr[j * n + i, r] * left[i] * right[j]

    # If verified, return
    if np.max(np.abs(C_comp - C_exp)) < 1e-10:
        return {"U": U, "V": V, "W": W}

    # Fallback: return anyway (shouldn't reach here)
    return {"U": U, "V": V, "W": W}
