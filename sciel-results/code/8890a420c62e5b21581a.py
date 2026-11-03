import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 110.0

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def verify_decomposition(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i * m + j]
                    v_val = V_mat[j * p + k]
                    w_val = W_mat[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    def to_half_int_list(mat):
        return [[int(round(mat[i, r] * 2)) / 2 for r in range(mat.shape[1])] for i in range(mat.shape[0])]

    # Strassen 2x2x2 patterns
    strassen_patterns = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    # Build rank-36 solution (verified working)
    total_rank_36 = 36
    U36 = np.zeros((nm, total_rank_36), dtype=np.float64)
    V36 = np.zeros((mp, total_rank_36), dtype=np.float64)
    W36 = np.zeros((pn, total_rank_36), dtype=np.float64)

    col = 0
    for k_block in range(2):  # k=0,1 and k=2,3
        for j_block in range(2):  # j=0,1 and j=2,3
            col = add_2x2x2_block(U36, V36, W36, 0, j_block * 2, j_block * 2, k_block * 2, 0, k_block * 2, col)
    # Add 2x2x1 for k=4
    for j_block in range(2):
        col = add_2x2x1_block(U36, V36, W36, 0, j_block * 2, j_block * 2, 4, 0, 4, col)

    assert col == total_rank_36, f"Expected 36, got {col}"

    if verify_decomposition(U36, V36, W36):
        best_result = {"U": to_half_int_list(U36), "V": to_half_int_list(V36), "W": to_half_int_list(W36)}
        best_rank = 36

    if time.time() > deadline - 20:
        return best_result

    # Try to find rank-34 using a rank-10 decomposition of 2x2x3
    # Strategy: split p=5 into p_left=3 and p_right=2
    # For each j-block (0-1 and 2-3), use 2x2x3 (rank 10) + 2x2x2 (rank 7) = 17
    # Total: 2 * 17 = 34
    
    # First, try to find a rank-10 decomposition of 2x2x3
    # The 2x2x3 multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We want: T = sum_{r=1}^{10} U_r[i] * V_r[j] * W_r[k]
    
    def find_2x2x3_rank10():
        """Find a rank-10 decomposition of the 2x2x3 multiplication tensor."""
        # Try the construction: 7 Strassen terms for k=0,1 + 3 terms for k=2
        sp = strassen_patterns
        
        terms = []
        for u_pat, v_pat, w_pat in sp:
            U = [u_pat[0], u_pat[1], u_pat[2], u_pat[3]]
            V = [v_pat[0], v_pat[1], v_pat[2], v_pat[3]]
            W = [w_pat[0], w_pat[1], w_pat[2], w_pat[3], 0]  # extend to 3
            terms.append((U, V, W))
        
        # Add 3 terms for k=2
        terms.append(([1, 0], [1, 0], [0, 0, 1]))
        terms.append(([0, 1], [1, 0], [0, 0, 1]))
        terms.append(([1, 1], [0, 1], [0, 0, 1]))
        
        # Verify
        T = np.zeros((2, 2, 3))
        for U, V, W in terms:
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        T[i, j, k] += U[i] * V[j] * W[k]
        
        if np.allclose(T, 1.0):
            return terms
        return None
    
    terms_223 = find_2x2x3_rank10()
    
    if terms_223 is not None and time.time() < deadline - 10:
        # Build the full 2x4x5 decomposition using 2x2x3 blocks
        total_rank_34 = 34
        U34 = np.zeros((nm, total_rank_34), dtype=np.float64)
        V34 = np.zeros((mp, total_rank_34), dtype=np.float64)
        W34 = np.zeros((pn, total_rank_34), dtype=np.float64)
        
        col = 0
        for j_block in range(2):
            # Add 2x2x3 terms for k=0,1,2
            for U_t, V_t, W_t in terms_223:
                # U_t[i] is the coefficient for A[i, j_local]
                # V_t[j] is the coefficient for B[j_local, k_local]
                # W_t[k] is the coefficient for C[i, k_local]
                for i in range(2):
                    U34[i * m + j_block * 2, col] = U_t[i]
                    U34[i * m + j_block * 2 + 1, col] = U_t[i]
                for j in range(2):
                    V34[(j_block * 2 + j) * p + 0, col] = V_t[j]
                    V34[(j_block * 2 + j) * p + 1, col] = V_t[j]
                    V34[(j_block * 2 + j) * p + 2, col] = V_t[j]
                for i in range(2):
                    W34[0 * n + i, col] = W_t[0]
                    W34[1 * n + i, col] = W_t[1]
                    W34[2 * n + i, col] = W_t[2]
                col += 1
            
            # Add 2x2x2 terms for k=3,4
            col = add_2x2x2_block(U34, V34, W34, 0, j_block * 2, j_block * 2, 3, 0, 3, col)
        
        assert col == total_rank_34, f"Expected 34, got {col}"
        
        if verify_decomposition(U34, V34, W34):
            best_result = {"U": to_half_int_list(U34), "V": to_half_int_list(V34), "W": to_half_int_list(W34)}
            best_rank = 34

    return best_result
# EVOLVE-BLOCK-END
