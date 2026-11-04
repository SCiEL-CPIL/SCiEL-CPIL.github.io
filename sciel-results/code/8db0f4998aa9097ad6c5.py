"""Matrix multiplication 6x6x6 tensor decomposition via Kronecker product of Strassen and schoolbook."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 100.0  # safety margin before 120s (evaluator kills at 150s)
    n, m, p = DIMENSIONS

    # Build 2x2 Strassen decomposition (rank 7)
    # U_str: 4x7, V_str: 4x7, W_str: 4x7
    U_s = np.zeros((4, 7), dtype=np.float64)
    V_s = np.zeros((4, 7), dtype=np.float64)
    W_s = np.zeros((4, 7), dtype=np.float64)

    # Strassen 7 terms for 2x2:
    # Term 0: (A00+A11)(B00+B11) -> C00, C11
    # Term 1: (A10+A11)B00 -> C10, C11
    # Term 2: A00(B01-B11) -> C01, C11
    # Term 3: A11(B10-B00) -> C00, C10
    # Term 4: (A00+A01)B11 -> C00, C01
    # Term 5: (A10-A00)(B00+B01) -> C11
    # Term 6: (A01-A11)(B10+B11) -> C00

    # U_s entries (row = a_idx*2+b_idx, col = term)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    U_s[2, 1] = 1; U_s[3, 1] = 1
    U_s[0, 2] = 1
    U_s[3, 3] = 1
    U_s[0, 4] = 1; U_s[1, 4] = 1
    U_s[2, 5] = 1; U_s[0, 5] = -1
    U_s[1, 6] = 1; U_s[3, 6] = -1

    # V_s entries
    V_s[0, 0] = 1; V_s[3, 0] = 1
    V_s[0, 1] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    V_s[3, 4] = 1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    V_s[2, 6] = 1; V_s[3, 6] = 1

    # W_s entries (row = c_idx*2+d_idx, col = term)
    W_s[0, 0] = 1; W_s[3, 0] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    W_s[3, 5] = 1
    W_s[0, 6] = 1

    # Build 3x3 schoolbook decomposition (rank 9)
    U_sc = np.zeros((9, 9), dtype=np.float64)
    V_sc = np.zeros((9, 9), dtype=np.float64)
    W_sc = np.zeros((9, 9), dtype=np.float64)
    for idx in range(9):
        U_sc[idx, idx] = 1
        V_sc[idx, idx] = 1
        W_sc[idx, idx] = 1

    # Kronecker product: rank = 7 * 9 = 63
    # U_total = kron(U_sc, U_s)  shape (36, 63)
    # V_total = kron(V_sc, V_s)  shape (36, 63)
    # W_total = kron(W_sc, W_s)  shape (36, 63)

    U_kron = np.kron(U_sc, U_s)  # (9*4, 9*7) = (36, 63)
    V_kron = np.kron(V_sc, V_s)  # (36, 63)
    W_kron = np.kron(W_sc, W_s)  # (36, 63)

    # We need to permute the rows to match the global index conventions.
    
    # U index: i (row of A), j (col of A)
    # Global A index: u = i * 6 + j
    # Block indices: i = I * 2 + i_sub, j = K * 2 + j_sub
    # u = (I * 2 + i_sub) * 6 + (K * 2 + j_sub) = I * 12 + i_sub * 6 + K * 2 + j_sub
    # Kron index: R = (I * 3 + K) * 4 + (i_sub * 2 + j_sub) = I * 12 + K * 4 + i_sub * 2 + j_sub
    
    def make_u_permutation():
        perm = np.zeros(36, dtype=int)
        for I in range(3):
            for K in range(3):
                for i_sub in range(2):
                    for j_sub in range(2):
                        kron_idx = (I * 3 + K) * 4 + (i_sub * 2 + j_sub)
                        desired_idx = (I * 2 + i_sub) * 6 + (K * 2 + j_sub)
                        perm[kron_idx] = desired_idx
        return perm

    perm_u = make_u_permutation()

    # V index: j (row of B), k (col of B)
    # Global B index: v = j * 6 + k
    # Block indices: j = K * 2 + j_sub, k = J * 2 + k_sub
    # v = (K * 2 + j_sub) * 6 + (J * 2 + k_sub) = K * 12 + j_sub * 6 + J * 2 + k_sub
    # Kron index: R = (K * 3 + J) * 4 + (j_sub * 2 + k_sub) = K * 12 + J * 4 + j_sub * 2 + k_sub

    def make_v_permutation():
        perm = np.zeros(36, dtype=int)
        for K in range(3):
            for J in range(3):
                for j_sub in range(2):
                    for k_sub in range(2):
                        kron_idx = (K * 3 + J) * 4 + (j_sub * 2 + k_sub)
                        desired_idx = (K * 2 + j_sub) * 6 + (J * 2 + k_sub)
                        perm[kron_idx] = desired_idx
        return perm

    perm_v = make_v_permutation()

    # W index: k (col of C), i (row of C)
    # Global C index: w = k * 6 + i
    # Block indices: k = J * 2 + k_sub, i = I * 2 + i_sub
    # w = (J * 2 + k_sub) * 6 + (I * 2 + i_sub) = J * 12 + k_sub * 6 + I * 2 + i_sub
    # Kron index: R = (I * 3 + J) * 4 + (i_sub * 2 + k_sub) = I * 12 + J * 4 + i_sub * 2 + k_sub

    def make_w_permutation():
        perm = np.zeros(36, dtype=int)
        for I in range(3):
            for J in range(3):
                for i_sub in range(2):
                    for k_sub in range(2):
                        kron_idx = (I * 3 + J) * 4 + (i_sub * 2 + k_sub)
                        desired_idx = (J * 2 + k_sub) * 6 + (I * 2 + i_sub)
                        perm[kron_idx] = desired_idx
        return perm

    perm_w = make_w_permutation()

    U_final = np.zeros((36, 63), dtype=np.float64)
    V_final = np.zeros((36, 63), dtype=np.float64)
    W_final = np.zeros((36, 63), dtype=np.float64)

    for kron_idx in range(36):
        U_final[perm_u[kron_idx], :] = U_kron[kron_idx, :]
        V_final[perm_v[kron_idx], :] = V_kron[kron_idx, :]
        W_final[perm_w[kron_idx], :] = W_kron[kron_idx, :]

    # Verify the decomposition
    if time.time() < deadline - 5:
        np.random.seed(42)
        A = np.random.randn(6, 6)
        B = np.random.randn(6, 6)
        C_expected = A @ B

        C_computed = np.zeros((6, 6))
        for i in range(6):
            for k in range(6):
                s = 0.0
                for r in range(63):
                    u_val = 0.0
                    for j in range(6):
                        u_val += A[i, j] * U_final[i * 6 + j, r]
                    v_val = 0.0
                    for j in range(6):
                        v_val += B[j, k] * V_final[j * 6 + k, r]
                    s += u_val * v_val * W_final[k * 6 + i, r]
                C_computed[i, k] = s

        err = np.max(np.abs(C_computed - C_expected))
        if err > 1e-9:
            # Fallback: use schoolbook 216
            rank_fb = 216
            U_fb = np.zeros((36, rank_fb))
            V_fb = np.zeros((36, rank_fb))
            W_fb = np.zeros((36, rank_fb))
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        idx = i * 36 + j * 6 + k
                        U_fb[i * 6 + j, idx] = 1
                        V_fb[j * 6 + k, idx] = 1
                        W_fb[k * 6 + i, idx] = 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
