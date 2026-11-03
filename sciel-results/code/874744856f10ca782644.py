import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = 2, 3, 8

    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

    # Rank-13 decomposition of M_{2,2,4}
    # A: 2x2, B: 2x4, C: 2x4
    # U: (4, 13), V: (8, 13), W: (8, 13)
    # U[i*2+j], V[j*4+k], W[k*2+i]
    def build_2x2x4_rank13():
        rank = 13
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(8)]

        # Term 0: (a00+a11)(b00+b12) -> c00, c12
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[6][0] = 1
        W[0][0] = 1; W[6][0] = 1

        # Term 1: (a10+a11)(b00+b10) -> c10, c11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1; V[4][1] = 1
        W[4][1] = 1; W[5][1] = 1

        # Term 2: a00(b01+b11) -> c01, c11
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = 1
        W[1][2] = 1; W[5][2] = 1

        # Term 3: (a00-a10)(b01+b10) -> c00, c10
        U[0][3] = 1; U[2][3] = -1
        V[1][3] = 1; V[4][3] = 1
        W[0][3] = 1; W[4][3] = 1

        # Term 4: (a01+a10)(b00-b10) -> c00, c01
        U[1][4] = 1; U[2][4] = 1
        V[0][4] = 1; V[4][4] = -1
        W[0][4] = -1; W[1][4] = 1

        # Term 5: (a11-a01)(b01-b11) -> c11
        U[1][5] = -1; U[3][5] = 1
        V[1][5] = 1; V[5][5] = -1
        W[5][5] = 1

        # Term 6: (a00+a01)(b11-b01) -> c00
        U[0][6] = 1; U[1][6] = 1
        V[1][6] = -1; V[5][6] = 1
        W[0][6] = 1

        # Term 7: (a00+a11)(b02+b12) -> c02, c12
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[6][7] = 1
        W[4][7] = 1; W[6][7] = 1

        # Term 8: (a10+a11)(b02+b12) -> c12, c13
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1; V[6][8] = 1
        W[6][8] = 1; W[7][8] = 1

        # Term 9: a00(b03+b13) -> c03, c13
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = 1
        W[5][9] = 1; W[7][9] = 1

        # Term 10: (a00-a10)(b03+b12) -> c02, c12
        U[0][10] = 1; U[2][10] = -1
        V[3][10] = 1; V[6][10] = 1
        W[4][10] = 1; W[6][10] = 1

        # Term 11: (a01+a10)(b02-b12) -> c02, c03
        U[1][11] = 1; U[2][11] = 1
        V[2][11] = 1; V[6][11] = -1
        W[4][11] = -1; W[5][11] = 1

        # Term 12: (a11-a01)(b03-b13) -> c13
        U[1][12] = -1; U[3][12] = 1
        V[3][12] = 1; V[7][12] = -1
        W[7][12] = 1

        return U, V, W

    # First, verify the rank-13 decomposition of M_{2,2,4}
    U13, V13, W13 = build_2x2x4_rank13()

    # Verify M_{2,2,4}: A(2x2) * B(2x4) = C(2x4)
    def verify_2x2x4(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # T[i,j,k] should be 1 for all i,j,k
        T = np.zeros((2, 2, 4))
        for c in range(r):
            u = U[:, c].reshape(2, 2)
            v = V[:, c].reshape(2, 4)
            w = W[:, c].reshape(4, 2)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((2, 2, 4)), atol=1e-10)

    if not verify_2x2x4(U13, V13, W13):
        # Fallback to rank-44 Strassen approach
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(2):
            for k in range(8):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        for block in range(4):
            b00 = 1 * p + block * 2
            b01 = 1 * p + block * 2 + 1
            b10 = 2 * p + block * 2
            b11 = 2 * p + block * 2 + 1
            c00 = (block * 2) * n + 0
            c01 = (block * 2 + 1) * n + 0
            c10 = (block * 2) * n + 1
            c11 = (block * 2 + 1) * n + 1
            U[1][col] = 1; U[5][col] = 1
            V[b00][col] = 1; V[b11][col] = 1
            W[c00][col] = 1; W[c11][col] = 1
            col += 1
            U[4][col] = 1; U[5][col] = 1
            V[b00][col] = 1
            W[c10][col] = 1; W[c11][col] = -1
            col += 1
            U[1][col] = 1
            V[b01][col] = 1; V[b11][col] = -1
            W[c01][col] = 1; W[c11][col] = 1
            col += 1
            U[5][col] = 1
            V[b10][col] = 1; V[b00][col] = -1
            W[c00][col] = 1; W[c10][col] = 1
            col += 1
            U[1][col] = 1; U[2][col] = 1
            V[b11][col] = 1
            W[c00][col] = -1; W[c01][col] = 1
            col += 1
            U[4][col] = 1; U[1][col] = -1
            V[b00][col] = 1; V[b01][col] = 1
            W[c11][col] = 1
            col += 1
            U[2][col] = 1; U[5][col] = -1
            V[b10][col] = 1; V[b11][col] = 1
            W[c00][col] = 1
            col += 1
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        # Final fallback: schoolbook
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col] = 1
                    V_f[j * p + k][col] = 1
                    W_f[k * n + i][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    # Build the full rank-42 decomposition for M_{2,3,8}
    # Part 1: 16 schoolbook terms for A[:,0] * B[0,:]
    # Part 2: 26 terms from two rank-13 M_{2,2,4} blocks
    total_rank = 16 + 26
    U = [[0] * total_rank for _ in range(n * m)]  # (6, 42)
    V = [[0] * total_rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * total_rank for _ in range(p * n)]  # (16, 42)

    col = 0
    # Part 1: A[:,0] (2x1) * B[0,:] (1x8) -> 2x8
    for i in range(2):
        for k in range(8):
            U[i * m + 0][col] = 1  # U[i*3+0]
            V[0 * p + k][col] = 1  # V[0*8+k]
            W[k * n + i][col] = 1  # W[k*2+i]
            col += 1

    # Part 2: Two rank-13 M_{2,2,4} blocks
    # Block 0: A[:,1:3] (2x2) * B[1:3, 0:4] (2x4) -> C[0:2, 0:4]
    # A indices: i*3+j where i=0,1 and j=1,2 -> U indices 1,2,4,5
    # B indices: j*8+k where j=1,2 and k=0,1,2,3 -> V indices 8,9,10,11,16,17,18,19
    # C indices: k*2+i where k=0,1,2,3 and i=0,1 -> W indices 0,1,2,3,4,5,6,7

    # Block 1: A[:,1:3] (2x2) * B[1:3, 4:8] (2x4) -> C[0:2, 4:8]
    # B indices: j*8+k where j=1,2 and k=4,5,6,7 -> V indices 12,13,14,15,20,21,22,23
    # C indices: k*2+i where k=4,5,6,7 and i=0,1 -> W indices 8,9,10,11,12,13,14,15

    for block in range(2):
        # Map local M_{2,2,4} indices to global indices
        # Local U: [a00, a01, a10, a11] -> global U: [1, 2, 4, 5]
        # Local V: [b00,b01,b02,b03,b10,b11,b12,b13]
        #   Block 0: [8, 9, 10, 11, 16, 17, 18, 19]
        #   Block 1: [12, 13, 14, 15, 20, 21, 22, 23]
        # Local W: [c00,c01,c02,c03,c10,c11,c12,c13]
        #   Block 0: [0, 1, 2, 3, 4, 5, 6, 7]
        #   Block 1: [8, 9, 10, 11, 12, 13, 14, 15]

        u_map = [1, 2, 4, 5]  # a00, a01, a10, a11
        if block == 0:
            v_map = [8, 9, 10, 11, 16, 17, 18, 19]
            w_map = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            v_map = [12, 13, 14, 15, 20, 21, 22, 23]
            w_map = [8, 9, 10, 11, 12, 13, 14, 15]

        for t in range(13):
            # Apply term t of the rank-13 decomposition
            for i_local in range(4):
                if U13[i_local][t] != 0:
                    U[u_map[i_local]][col] += U13[i_local][t]
            for j_local in range(8):
                if V13[j_local][t] != 0:
                    V[v_map[j_local]][col] += V13[j_local][t]
            for k_local in range(8):
                if W13[k_local][t] != 0:
                    W[w_map[k_local]][col] += W13[k_local][t]
            col += 1

    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: rank-44 Strassen approach
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(2):
        for k in range(8):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    for block in range(4):
        b00 = 1 * p + block * 2
        b01 = 1 * p + block * 2 + 1
        b10 = 2 * p + block * 2
        b11 = 2 * p + block * 2 + 1
        c00 = (block * 2) * n + 0
        c01 = (block * 2 + 1) * n + 0
        c10 = (block * 2) * n + 1
        c11 = (block * 2 + 1) * n + 1
        U[1][col] = 1; U[5][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        U[4][col] = 1; U[5][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1; W[c11][col] = -1
        col += 1
        U[1][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1
        U[5][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1
        U[1][col] = 1; U[2][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1; W[c01][col] = 1
        col += 1
        U[4][col] = 1; U[1][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        U[2][col] = 1; U[5][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1
        col += 1
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Final fallback: schoolbook
    rank_fallback = n * m * p
    U_f = [[0] * rank_fallback for _ in range(n * m)]
    V_f = [[0] * rank_fallback for _ in range(m * p)]
    W_f = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j][col] = 1
                V_f[j * p + k][col] = 1
                W_f[k * n + i][col] = 1
                col += 1
    return {"U": U_f, "V": V_f, "W": W_f}
# EVOLVE-BLOCK-END
