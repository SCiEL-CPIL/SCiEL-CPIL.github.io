import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Helper to get Schoolbook 3x3 decomposition (Rank 27)
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

    # Helper to get Rank-23 3x3 decomposition
    # Note: Hardcoding a specific rank-23 decomposition is complex and error-prone without a reference.
    # We will attempt to use a known sparse structure or fall back to schoolbook.
    # For safety, we use Schoolbook (27) as the primary kernel, but we can try to optimize.
    # However, to ensure a jump, we need a lower rank.
    # Let's use a Rank-23 decomposition if we can verify it.
    # Since I cannot guarantee the coefficients, I will use the Schoolbook 27.
    # To improve the score, I will ensure the Strassen composition is correct and verified.
    
    U3, V3, W3 = get_3x3_schoolbook()
    r3 = 27

    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    def add_strassen_term(start_col, a_blocks, b_blocks, c_blocks):
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

    # Strassen formulas
    add_strassen_term(0, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
    add_strassen_term(r3, [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])
    add_strassen_term(2 * r3, [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])
    add_strassen_term(3 * r3, [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])
    add_strassen_term(4 * r3, [(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])
    add_strassen_term(5 * r3, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    add_strassen_term(6 * r3, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])

    # Verify
    if time.time() < deadline:
        try:
            U3g = U6.reshape(6, 6, -1)
            V3g = V6.reshape(6, 6, -1)
            W3g = W6.reshape(6, 6, -1)
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U3g, V3g, W3g)
            if np.allclose(reconstructed, np.ones((6, 6, 6)), atol=1e-9):
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        except Exception:
            pass

    # Fallback: Schoolbook 6x6
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
