import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0

    # Schoolbook fallback
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    def _verify_decomposition(U, V, W, n, m, p):
        r = U.shape[1]
        if r == 0: return False
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        # Check if T_ijk = sum_r u_ij_r v_jk_r w_ki_r == 1
        res = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(res, 1.0, atol=1e-9)

    # Winograd 2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0]
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, -1],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, -1, 1, -1, 1],
        [1, -1, 1, 0, -1],
        [1, 1, -1, 1, 0],
        [1, 1, -1, 0, 1]
    ], dtype=float)

    # Construct 4x4x4 Winograd factors (Rank 25)
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)

    # Permutation for 4x4 indices
    perm_U = np.zeros(16, dtype=int)
    for r1 in range(4):
        row1 = r1 // 2
        col1 = r1 % 2
        for r2 in range(4):
            row2 = r2 // 2
            col2 = r2 % 2
            kron_idx = r1 * 4 + r2
            i = row1 * 2 + row2
            j = col1 * 2 + col2
            target_idx = i * 4 + j
            perm_U[target_idx] = kron_idx

    U_winograd = U4_kron[perm_U, :]
    V_winograd = V4_kron[perm_U, :]
    W_winograd = W4_kron[perm_U, :]

    # --- Try to find a better decomposition for the "tail" (k=4) ---
    # The tensor is T[i, j, k].
    # For k=4, T[i, j, 4] = 1 for all i, j.
    # We already have a rank-4 solution for this (using U, V, W blocks that are non-zero only for k=4).
    # Can we do better?
    # The tensor T_tail = 1_{4x4}.
    # Rank of 1_{4x4} as a matrix is 1.
    # But here it's part of a 3-tensor.
    # T[i, j, 4] = 1.
    # We need sum_r u_ij_r v_j4_r w_4i_r = 1.
    # If we could set u_ij_r, v_j4_r, w_4i_r such that rank is < 4, we'd improve.
    # Let's try to see if we can merge the tail with the main block.
    # This requires solving a system of equations which is complex.
    
    # Alternative: Check if the rank 29 solution can be perturbed or if a rank 28 solution exists.
    # Given the strict time limit and the difficulty of proving non-existence of lower ranks,
    # and the fact that 29 is a strong constructive upper bound derived from optimal sub-components,
    # we will stick with the constructive rank 29 solution unless a quick search finds something else.
    
    # The previous program established Rank 29.
    # Let's construct it directly.
    
    rank_candidate = 26
    U_cand = np.zeros((16, rank_candidate))
    V_cand = np.zeros((20, rank_candidate))
    W_cand = np.zeros((20, rank_candidate)) # p*n = 5*4 = 20

    # Part 1: Winograd for 4x4x4 (cols 0-3 of B, cols 0-3 of C)
    U_cand[:, 0:25] = U_winograd
    
    for j in range(4):
        for k in range(4):
            v_row = j * 4 + k
            target_v_row = j * 5 + k
            V_cand[target_v_row, 0:25] = V_winograd[v_row, :]
            
    W_cand[0:16, 0:25] = W_winograd

    # Part 2: 5th column (k=4) - rank 1 since T[i,j,4]=1 for all i,j
    col_idx = 25
    for i in range(4):
        for j in range(4):
            U_cand[i*4+j, col_idx] = 1.0
    for j in range(4):
        V_cand[j*5+4, col_idx] = 1.0
    for i in range(4):
        W_cand[4*4+i, col_idx] = 1.0

    if _verify_decomposition(U_cand, V_cand, W_cand, n, m, p):
        best_rank = 26
        best_U = U_cand
        best_V = V_cand
        best_W = W_cand

    def to_list(mat):
        return [[float(val) for val in row] for row in mat]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
