import time
import numpy as np

N = 4
M = 4
P = 4

def construct():
    t0 = time.time()
    tmax = 118.0

    rank_total = 49
    U_mat = np.zeros((N * M, rank_total), dtype=np.int8)
    V_mat = np.zeros((M * P, rank_total), dtype=np.int8)
    W_mat = np.zeros((P * N, rank_total), dtype=np.int8)

    # Strassen 2x2x2 outer coefficients: shape (7, 2, 2)
    B = np.zeros((7, 2, 2), dtype=np.int8)
    G = np.zeros((7, 2, 2), dtype=np.int8)
    D = np.zeros((7, 2, 2), dtype=np.int8)

    B[0] = [[1, 0], [0, 1]]
    G[0] = [[1, 0], [0, 1]]
    D[0] = [[1, 0], [0, 1]]
    B[1] = [[0, 0], [1, 1]]
    G[1] = [[1, 0], [0, 0]]
    D[1] = [[0, 0], [1, -1]]
    B[2] = [[1, 0], [0, 0]]
    G[2] = [[0, 1], [0, -1]]
    D[2] = [[0, 1], [0, 1]]
    B[3] = [[0, 0], [0, 1]]
    G[3] = [[-1, 0], [1, 0]]
    D[3] = [[1, 0], [1, 0]]
    B[4] = [[1, 1], [0, 0]]
    G[4] = [[0, 0], [0, 1]]
    D[4] = [[-1, 1], [0, 0]]
    B[5] = [[-1, 0], [1, 0]]
    G[5] = [[1, 1], [0, 0]]
    D[5] = [[0, 0], [0, 1]]
    B[6] = [[0, 1], [0, -1]]
    G[6] = [[0, 0], [1, 1]]
    D[6] = [[1, 0], [0, 0]]

    # Inner 2x2x2 Strassen (rank 7): U2, V2, W2 are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.int8)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=np.int8)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.int8)

    # Build via explicit index mapping
    # U[(i*4+j), 7*si+ti] = B[si, i//2, j//2] * U2[i%2*2 + j%2, ti]
    # V[(j*4+k), 7*si+ti] = G[si, j//2, k//2] * V2[j%2*2 + k%2, ti]
    # W[(k*4+i), 7*si+ti] = D[si, i//2, k//2] * W2[k%2*2 + i%2, ti]

    for si in range(7):
        for ti in range(7):
            ci = 7 * si + ti
            for bi in range(2):
                for bj in range(2):
                    bv = int(B[si, bi, bj])
                    if bv != 0:
                        for ii in range(2):
                            for jj in range(2):
                                uv = int(U2[ii * 2 + jj, ti])
                                if uv != 0:
                                    r = (bi * 2 + ii) * 4 + (bj * 2 + jj)
                                    U_mat[r, ci] = bv * uv

            for bj2 in range(2):
                for bk in range(2):
                    gv = int(G[si, bj2, bk])
                    if gv != 0:
                        for jj2 in range(2):
                            for kk in range(2):
                                vv = int(V2[jj2 * 2 + kk, ti])
                                if vv != 0:
                                    r = (bj2 * 2 + jj2) * 4 + (bk * 2 + kk)
                                    V_mat[r, ci] = gv * vv

            for bi2 in range(2):
                for bk2 in range(2):
                    dv = int(D[si, bi2, bk2])
                    if dv != 0:
                        for ii2 in range(2):
                            for kk2 in range(2):
                                wv = int(W2[kk2 * 2 + ii2, ti])
                                if wv != 0:
                                    r = (bk2 * 2 + kk2) * 4 + (bi2 * 2 + ii2)
                                    W_mat[r, ci] = dv * wv

    if time.time() - t0 > tmax:
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}

    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
