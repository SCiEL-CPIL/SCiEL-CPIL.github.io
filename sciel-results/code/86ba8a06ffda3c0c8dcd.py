import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # 2x2 Strassen decomposition (rank 7)
    def get_2x2_strassen():
        n2 = 2
        rank2 = 7
        U2 = np.zeros((4, rank2))
        V2 = np.zeros((4, rank2))
        W2 = np.zeros((4, rank2))
        # Blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # P1 = (A00+A11)*(B00+B11) -> C00(+1), C11(+1)
        for r in range(rank2):
            if r == 0:
                U2[0, r] = 1; U2[3, r] = 1
                V2[0, r] = 1; V2[3, r] = 1
                W2[0, r] = 1; W2[3, r] = 1
            elif r == 1:
                U2[2, r] = 1; U2[3, r] = 1
                V2[0, r] = 1
                W2[2, r] = 1; W2[3, r] = -1
            elif r == 2:
                U2[0, r] = 1
                V2[1, r] = 1; V2[3, r] = -1
                W2[1, r] = 1; W2[3, r] = 1
            elif r == 3:
                U2[3, r] = 1
                V2[2, r] = 1; V2[0, r] = -1
                W2[0, r] = 1; W2[2, r] = 1
            elif r == 4:
                U2[0, r] = 1; U2[1, r] = 1
                V2[3, r] = 1
                W2[1, r] = 1; W2[0, r] = -1
            elif r == 5:
                U2[2, r] = 1; U2[0, r] = -1
                V2[0, r] = 1; V2[1, r] = 1
                W2[3, r] = 1
            elif r == 6:
                U2[1, r] = 1; U2[3, r] = -1
                V2[2, r] = 1; V2[3, r] = 1
                W2[0, r] = 1
        return U2, V2, W2

    U2, V2, W2 = get_2x2_strassen()
    r2 = 7

    # 3x3 schoolbook decomposition (rank 27)
    def get_3x3_schoolbook():
        n3 = 3
        rank3 = 27
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    col = i * 9 + j * 3 + k
                    U3[i * 3 + j, col] = 1.0
                    V3[j * 3 + k, col] = 1.0
                    W3[k * 3 + i, col] = 1.0
        return U3, V3, W3

    U3, V3, W3 = get_3x3_schoolbook()
    r3 = 27

    # Approach: Strassen on 2x2 blocks of 3x3 matrices.
    # This gives rank 7 * 27 = 189.
    # The key is to correctly map the 3x3 decomposition terms into the 6x6 structure.
    # For each Strassen product P_k = M_k * N_k where M_k and N_k are 3x3 linear combos,
    # we use the 3x3 decomposition to compute M_k * N_k.
    
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    def add_strassen_term(start_col, a_blocks, b_blocks, c_blocks):
        """
        Add one Strassen term to the global decomposition.
        a_blocks: list of (block_idx, coeff) for A
        b_blocks: list of (block_idx, coeff) for B
        c_blocks: list of (block_idx, coeff) for C
        Block idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        """
        for r in range(r3):
            u_mat = U3[:, r].reshape(3, 3)
            v_mat = V3[:, r].reshape(3, 3)
            w_mat = W3[:, r].reshape(3, 3)

            for bi, coeff in a_blocks:
                rb, cb = bi // 2, bi % 2
                for i in range(3):
                    for j in range(3):
                        val = coeff * u_mat[i, j]
                        if val != 0:
                            gi, gj = rb * 3 + i, cb * 3 + j
                            U6[gi * 6 + gj, start_col + r] += val

            for bi, coeff in b_blocks:
                rb, cb = bi // 2, bi % 2
                for j in range(3):
                    for k in range(3):
                        val = coeff * v_mat[j, k]
                        if val != 0:
                            gj, gk = rb * 3 + j, cb * 3 + k
                            V6[gj * 6 + gk, start_col + r] += val

            for bi, coeff in c_blocks:
                rb, cb = bi // 2, bi % 2
                for k in range(3):
                    for i in range(3):
                        val = coeff * w_mat[k, i]
                        if val != 0:
                            gk, gi = cb * 3 + k, rb * 3 + i
                            W6[gk * 6 + gi, start_col + r] += val

    # Strassen formulas for 2x2 blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # P1 = (A11 + A22) * (B11 + B22) -> C11, C22
    add_strassen_term(0,
                      a_blocks=[(0, 1), (3, 1)],
                      b_blocks=[(0, 1), (3, 1)],
                      c_blocks=[(0, 1), (3, 1)])

    # P2 = (A21 + A22) * B11 -> C21 (+1), C22 (-1)
    add_strassen_term(r3,
                      a_blocks=[(2, 1), (3, 1)],
                      b_blocks=[(0, 1)],
                      c_blocks=[(2, 1), (3, -1)])

    # P3 = A11 * (B12 - B22) -> C12 (+1), C22 (+1)
    add_strassen_term(2 * r3,
                      a_blocks=[(0, 1)],
                      b_blocks=[(1, 1), (3, -1)],
                      c_blocks=[(1, 1), (3, 1)])

    # P4 = A22 * (B21 - B11) -> C11 (+1), C21 (+1)
    add_strassen_term(3 * r3,
                      a_blocks=[(3, 1)],
                      b_blocks=[(2, 1), (0, -1)],
                      c_blocks=[(0, 1), (2, 1)])

    # P5 = (A11 + A12) * B22 -> C12 (+1), C11 (-1)
    add_strassen_term(4 * r3,
                      a_blocks=[(0, 1), (1, 1)],
                      b_blocks=[(3, 1)],
                      c_blocks=[(1, 1), (0, -1)])

    # P6 = (A21 - A11) * (B11 + B12) -> C22 (+1)
    add_strassen_term(5 * r3,
                      a_blocks=[(2, 1), (0, -1)],
                      b_blocks=[(0, 1), (1, 1)],
                      c_blocks=[(3, 1)])

    # P7 = (A12 - A22) * (B21 + B22) -> C11 (+1)
    add_strassen_term(6 * r3,
                      a_blocks=[(1, 1), (3, -1)],
                      b_blocks=[(2, 1), (3, 1)],
                      c_blocks=[(0, 1)])

    # Verify the decomposition
    try:
        U_mat = U6.reshape(6, 6, -1)  # [i, j, r]
        V_mat = V6.reshape(6, 6, -1)  # [j, k, r]
        W_mat = W6.reshape(6, 6, -1)  # [k, i, r]
        
        T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
        expected = np.ones((6, 6, 6))
        if np.allclose(T, expected, atol=1e-9):
            return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook rank 216
    rank = 216
    U_f = np.zeros((36, rank))
    V_f = np.zeros((36, rank))
    W_f = np.zeros((36, rank))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                col = i * 36 + j * 6 + k
                U_f[i * 6 + j, col] = 1.0
                V_f[j * 6 + k, col] = 1.0
                W_f[k * 6 + i, col] = 1.0
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
