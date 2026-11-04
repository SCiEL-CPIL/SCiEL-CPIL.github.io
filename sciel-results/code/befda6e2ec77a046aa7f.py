import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 130.0

def construct():
    n, m, p = 4, 4, 8
    deadline = time.time() + 130.0

    # Strassen 2x2x2 decomposition (rank 7)
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7), dtype=np.int8)
    V2 = np.zeros((4, 7), dtype=np.int8)
    W2 = np.zeros((4, 7), dtype=np.int8)

    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build 4x4x4 decomposition via tensor product (rank 49)
    U4 = np.zeros((16, 49), dtype=np.int16)
    V4 = np.zeros((16, 49), dtype=np.int16)
    W4 = np.zeros((16, 49), dtype=np.int16)

    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    U4[i*4+j, c] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]

            for j in range(4):
                for k in range(4):
                    j0, j1 = j // 2, j % 2
                    k0, k1 = k // 2, k % 2
                    V4[j*4+k, c] = V2[j0*2+k0, c0] * V2[j1*2+k1, c1]

            for k in range(4):
                for i in range(4):
                    k0, k1 = k // 2, k % 2
                    i0, i1 = i // 2, i % 2
                    W4[k*4+i, c] = W2[k0*2+i0, c0] * W2[k1*2+i1, c1]

    # Build 4x4x8 decomposition (rank 98)
    rank = 98
    U = np.zeros((16, rank), dtype=np.int16)
    V = np.zeros((32, rank), dtype=np.int16)
    W = np.zeros((32, rank), dtype=np.int16)

    for b in range(2):
        col_start = b * 49
        col_end = (b + 1) * 49

        U[:, col_start:col_end] = U4

        for j in range(4):
            for k in range(4):
                k_actual = b * 4 + k
                V[j*8 + k_actual, col_start:col_end] = V4[j*4 + k, :]

        for k in range(4):
            k_actual = b * 4 + k
            for i in range(4):
                W[k_actual*4 + i, col_start:col_end] = W4[k*4 + i, :]

    # Vectorized verification using einsum
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape to 3D tensors and use einsum
    U3 = U.reshape(n, m, rank).astype(np.float64)
    V3 = V.reshape(m, p, rank).astype(np.float64)
    W3 = W.reshape(p, n, rank).astype(np.float64)

    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kr->ijk', U3, V3, W3)

    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=1e-9):
        # Fallback: schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=np.int8)
        V_sb = np.zeros((m*p, rank_sb), dtype=np.int8)
        W_sb = np.zeros((p*n, rank_sb), dtype=np.int8)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
