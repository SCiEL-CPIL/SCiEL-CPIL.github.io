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

    # Strassen-based decomposition for 2x4x5
    # Rank 36
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        strassen_patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
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

    col = 0
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)
    assert col == total_rank

    def verify_decomposition(U, V, W):
        rank = U.shape[1]
        # Check diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check off-diagonal terms
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
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    # Try to find rank-35 by merging a Strassen column with a schoolbook column
    # Strategy: for each pair (s, x) where s is a Strassen column and x is a schoolbook column,
    # try to find a scalar alpha such that the merged column preserves the decomposition.
    # 
    # The key insight: if we have a rank-36 decomposition and want rank 35, we need to find
    # a linear dependence among the columns that preserves the tensor structure.
    # 
    # For 2x4x5, the known rank-35 decomposition can be obtained by a specific merge.
    # Let's try: merge the last schoolbook column of the first 2x2x1 block with the first
    # Strassen column of the second 2x2x2 block.
    
    # The schoolbook columns in the first 2x2x1 block are at positions 14,15,16,17
    # The Strassen columns in the second 2x2x2 block are at positions 18-24
    
    # Try merging schoolbook column 17 (i=1, j=3, k=4) with Strassen column 18
    # This is a heuristic based on the structure of the known rank-35 decomposition.
    
    found_rank35 = False
    U35 = None
    V35 = None
    W35 = None
    
    if time.time() < deadline - 30:
        # Try various merge strategies
        # For each schoolbook column in the 2x2x1 blocks, try merging with each Strassen column
        # in adjacent 2x2x2 blocks
        
        # Schoolbook columns: 14,15,16,17 (first block), 32,33,34,35 (second block)
        # Strassen columns: 0-6, 7-13 (first 2x2x2 blocks), 18-24, 25-31 (second 2x2x2 blocks)
        
        schoolbook_cols = [14, 15, 16, 17, 32, 33, 34, 35]
        strassen_cols = list(range(0, 7)) + list(range(7, 14)) + list(range(18, 25)) + list(range(25, 32))
        
        # For each pair, try alpha values
        alphas = [0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75, -0.75]
        
        for sc in schoolbook_cols:
            for tc in strassen_cols:
                for alpha in alphas:
                    if time.time() > deadline - 10:
                        break
                    # Create a copy and merge: new_col = alpha * col_tc + col_sc
                    U_test = U.copy()
                    V_test = V.copy()
                    W_test = W.copy()
                    
                    # Merge column tc into sc: new_sc = col_sc + alpha * col_tc
                    U_test[:, sc] = U_test[:, sc] + alpha * U_test[:, tc]
                    V_test[:, sc] = V_test[:, sc] + alpha * V_test[:, tc]
                    W_test[:, sc] = W_test[:, sc] + alpha * W_test[:, tc]
                    
                    # Remove column tc
                    U_test = np.delete(U_test, tc, axis=1)
                    V_test = np.delete(V_test, tc, axis=1)
                    W_test = np.delete(W_test, tc, axis=1)
                    
                    if verify_decomposition(U_test, V_test, W_test):
                        found_rank35 = True
                        U35 = U_test
                        V35 = V_test
                        W35 = W_test
                        break
                if found_rank35 or time.time() > deadline - 10:
                    break
            if found_rank35 or time.time() > deadline - 10:
                break
        
        if found_rank35:
            # Round to nearest half-integer
            U35_list = [[int(round(U35[i, r] * 2)) / 2 for r in range(35)] for i in range(nm)]
            V35_list = [[int(round(V35[i, r] * 2)) / 2 for r in range(35)] for i in range(mp)]
            W35_list = [[int(round(W35[i, r] * 2)) / 2 for r in range(35)] for i in range(pn)]
            return {"U": U35_list, "V": V35_list, "W": W35_list}

    # If rank 35 not found, return rank 36
    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return best_result
# EVOLVE-BLOCK-END
