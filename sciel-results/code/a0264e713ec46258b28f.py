import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if time.time() - t0 > 135.0:
        return schoolbook()

    # 2x2 Strassen Matrices (Rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Verify 2x2 Strassen correctness
    valid_2x2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                if s != 1:
                    valid_2x2 = False
                    break
            if not valid_2x2:
                break
        if not valid_2x2:
            break

    if not valid_2x2:
        return schoolbook()

    # Construct 4x4 decomposition via Kronecker product
    # Rank becomes 7 * 7 = 49
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)

    # Permute rows to match the linear index convention:
    # U row index: i*m + j = i*4 + j
    # V row index: j*p + k = j*4 + k
    # W row index: k*n + i = k*4 + i

    perm = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4

        i_out = i // 2
        i_in = i % 2
        j_out = j // 2
        j_in = j % 2

        row_A = i_out * 2 + j_out
        row_B = i_in * 2 + j_in
        R = row_A * 4 + row_B

        perm[S] = R

    U4 = U4_kron[perm, :]
    V4 = V4_kron[perm, :]
    W4 = W4_kron[perm, :]

    # Verify 4x4 correctness
    valid_4x4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_row = U4[i*4+j, :]
                v_row = V4[j*4+k, :]
                w_row = W4[k*4+i, :]
                s = np.dot(u_row, v_row * w_row)
                if s != 1:
                    valid_4x4 = False
                    break
            if not valid_4x4:
                break
        if not valid_4x4:
            break

    if not valid_4x4:
        return schoolbook()

    if time.time() - t0 > 140.0:
        return schoolbook()

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
