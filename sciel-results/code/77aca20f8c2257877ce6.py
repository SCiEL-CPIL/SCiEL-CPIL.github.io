import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 90

    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2   # 8 blocks of width 2
    rank = 7 * num_blocks # 56

    # Strassen 2x2x2 decomposition: 7 rank-1 terms
    # U: coefficients for A (2x2), indexed as [A00, A01, A10, A11]
    # V: coefficients for B (2x2), indexed as [B00, B01, B10, B11]
    # W: coefficients for C (2x2), indexed as [C00, C01, C10, C11]
    #
    # Standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6

    strassen_U = np.array([
        [1, 0, 0, 1],  # M1: a00+a11
        [0, 0, 1, 1],  # M2: a10+a11
        [1, 0, 0, 0],  # M3: a00
        [0, 0, 0, 1],  # M4: a11
        [1, 1, 0, 0],  # M5: a00+a01
        [-1, 0, 1, 0], # M6: a10-a00
        [0, 1, 0, -1], # M7: a01-a11
    ], dtype=float)

    strassen_V = np.array([
        [1, 0, 0, 1],  # M1: b00+b11
        [1, 0, 0, 0],  # M2: b00
        [0, 1, 0, -1], # M3: b01-b11
        [-1, 0, 1, 0], # M4: b10-b00
        [0, 0, 0, 1],  # M5: b11
        [1, 1, 0, 0],  # M6: b00+b01
        [0, 0, 1, 1],  # M7: b10+b11
    ], dtype=float)

    strassen_W = np.array([
        [1, 0, 0, 1],  # M1: c00+1, c11+1
        [0, 0, 1, -1], # M2: c10+1, c11-1
        [0, 1, 0, 1],  # M3: c01+1, c11+1
        [1, 0, 1, 0],  # M4: c00+1, c10+1
        [-1, 1, 0, 0], # M5: c00-1, c01+1
        [0, 0, 0, 1],  # M6: c11+1
        [1, 0, 0, 0],  # M7: c00+1
    ], dtype=float)

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for block_idx in range(num_blocks):
        if time.time() > deadline:
            break
        k_base = 2 * block_idx
        for term_idx in range(7):
            col = block_idx * 7 + term_idx

            # U: same for all blocks, index is (i, j) -> row i*m + j
            for i in range(n):
                for j in range(m):
                    U[i * m + j, col] = strassen_U[term_idx, i * m + j]

            # V: local B indices mapped to global
            # B00 -> row 0*p + k_base, B01 -> row 0*p + k_base+1
            # B10 -> row 1*p + k_base, B11 -> row 1*p + k_base+1
            for j in range(m):
                for k_local in range(2):
                    k = k_base + k_local
                    V[j * p + k, col] = strassen_V[term_idx, j * 2 + k_local]

            # W: local C indices mapped to global
            # C00 -> row k_base*n + 0, C01 -> row k_base*n + 1
            # C10 -> row (k_base+1)*n + 0, C11 -> row (k_base+1)*n + 1
            for k_local in range(2):
                for i in range(n):
                    k = k_base + k_local
                    W[k * n + i, col] = strassen_W[term_idx, k_local * n + i]

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        rank_val = U_mat.shape[1]
        # Check valid terms: T[i*m+j, j*p+k, k*n+i] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0.0
                    for s in range(rank_val):
                        val += U_mat[u_idx, s] * V_mat[v_idx, s] * W_mat[w_idx, s]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check that no spurious terms exist for invalid index combinations
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i * m + j, j * p + k, k * n + i))
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in valid_set:
                        val = 0.0
                        for s in range(rank_val):
                            val += U_mat[a, s] * V_mat[b, s] * W_mat[c, s]
                        if abs(val) > 1e-9:
                            return False
        return True

    if time.time() < deadline - 5:
        if verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook decomposition (rank 64)
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school), dtype=float)
    V_s = np.zeros((m * p, rank_school), dtype=float)
    W_s = np.zeros((p * n, rank_school), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1.0
                V_s[j * p + k, col] = 1.0
                W_s[k * n + i, col] = 1.0
                col += 1

    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
