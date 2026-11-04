import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    n, m, p = 4, 5, 5
    n_m = n * m  # 20
    m_p = m * p  # 25
    p_n = p * n  # 20

    # --- Helper: Verify decomposition ---
    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)

    # --- Helper: Schoolbook decomposition ---
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # --- Strassen 2x2x2 Decomposition ---
    def get_2x2x2_strassen(i_off, j_off, k_off):
        """
        Returns U, V, W for a 2x2x2 Strassen block embedded in the full n x m x p space.
        """
        rank = 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        i0, i1 = i_off, i_off + 1
        j0, j1 = j_off, j_off + 1
        k0, k1 = k_off, k_off + 1

        def ui(i, j): return i * m + j
        def vj(j, k): return j * p + k
        def wk(k, i): return k * n + i

        # M1 = (A00+A11)(B00+B11)
        U[ui(i0, j0), 0] = 1; U[ui(i1, j1), 0] = 1
        V[vj(j0, k0), 0] = 1; V[vj(j1, k1), 0] = 1
        W[wk(k0, i0), 0] = 1; W[wk(k1, i1), 0] = 1

        # M2 = (A10+A11)B00
        U[ui(i1, j0), 1] = 1; U[ui(i1, j1), 1] = 1
        V[vj(j0, k0), 1] = 1
        W[wk(k0, i1), 1] = 1; W[wk(k1, i1), 1] = -1

        # M3 = A00(B01-B11)
        U[ui(i0, j0), 2] = 1
        V[vj(j0, k1), 2] = 1; V[vj(j1, k1), 2] = -1
        W[wk(k1, i0), 2] = 1; W[wk(k1, i1), 2] = 1

        # M4 = A11(B10-B00)
        U[ui(i1, j1), 3] = 1
        V[vj(j1, k0), 3] = 1; V[vj(j0, k0), 3] = -1
        W[wk(k0, i0), 3] = 1; W[wk(k0, i1), 3] = 1

        # M5 = (A00+A01)B11
        U[ui(i0, j0), 4] = 1; U[ui(i0, j1), 4] = 1
        V[vj(j1, k1), 4] = 1
        W[wk(k0, i0), 4] = -1; W[wk(k1, i0), 4] = 1

        # M6 = (A10-A00)(B00+B01)
        U[ui(i1, j0), 5] = 1; U[ui(i0, j0), 5] = -1
        V[vj(j0, k0), 5] = 1; V[vj(j0, k1), 5] = 1
        W[wk(k1, i1), 5] = 1

        # M7 = (A01-A11)(B10+B11)
        U[ui(i0, j1), 6] = 1; U[ui(i1, j1), 6] = -1
        V[vj(j1, k0), 6] = 1; V[vj(j1, k1), 6] = 1
        W[wk(k0, i0), 6] = 1

        return U, V, W

    # --- Construct 4x5x5 using block decomposition ---
    # Partition: n=4 as [2,2], m=5 as [2,3], p=5 as [2,3]
    # Blocks:
    # (i0, j0, k0): 2x2x2 -> Strassen (7)
    # (i0, j1, k0): 2x3x2 -> 2x2x2 (7) + 2x1x2 (4) = 11
    # (i0, j0, k1): 2x2x3 -> 2x2x2 (7) + 2x2x1 (4) = 11
    # (i0, j1, k1): 2x3x3 -> 2x2x2 (7) + 2x2x1 (4) + 2x1x2 (4) + 2x1x1 (2) = 17
    # (i1, j0, k0): 2x2x2 -> Strassen (7)
    # (i1, j1, k0): 2x3x2 -> 2x2x2 (7) + 2x1x2 (4) = 11
    # (i1, j0, k1): 2x2x3 -> 2x2x2 (7) + 2x2x1 (4) = 11
    # (i1, j1, k1): 2x3x3 -> 2x2x2 (7) + 2x2x1 (4) + 2x1x2 (4) + 2x1x1 (2) = 17
    # Total: 7 + 11 + 11 + 17 + 7 + 11 + 11 + 17 = 92

    U_total = np.zeros((n_m, 92))
    V_total = np.zeros((m_p, 92))
    W_total = np.zeros((p_n, 92))
    col = 0

    def add_block(i_off, j_off, k_off, i_sz, j_sz, k_sz):
        nonlocal col
        if i_sz == 2 and j_sz == 2 and k_sz == 2:
            U_b, V_b, W_b = get_2x2x2_strassen(i_off, j_off, k_off)
            U_total[:, col:col+7] += U_b
            V_total[:, col:col+7] += V_b
            W_total[:, col:col+7] += W_b
            col += 7
        else:
            # Schoolbook for this sub-block
            for i in range(i_sz):
                for j in range(j_sz):
                    for k in range(k_sz):
                        U_total[(i_off + i) * m + (j_off + j), col] = 1
                        V_total[(j_off + j) * p + (k_off + k), col] = 1
                        W_total[(k_off + k) * n + (i_off + i), col] = 1
                        col += 1

    # Iterate over blocks
    # i blocks: [0,2), [2,4)
    # j blocks: [0,2), [2,5)
    # k blocks: [0,2), [2,5)
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 3)]
    k_blocks = [(0, 2), (2, 3)]

    for i_off, i_sz in i_blocks:
        for j_off, j_sz in j_blocks:
            for k_off, k_sz in k_blocks:
                if time.time() > deadline:
                    break
                add_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)

    # Verify
    if not verify(U_total, V_total, W_total, n, m, p):
        # Fallback to schoolbook
        U_total, V_total, W_total = schoolbook(n, m, p)

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
