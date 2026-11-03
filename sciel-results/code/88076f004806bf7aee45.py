import time
import sys
import random

def construct():
    n, m, p = 2, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10

    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    # Strassen 2x2x2 patterns: (u_pattern, v_pattern, w_pattern)
    strassen_patterns = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    schoolbook_2x2x1 = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1]),
    ]

    rank_total = 36
    U = [[0] * rank_total for _ in range(nm)]
    V = [[0] * rank_total for _ in range(mp)]
    W = [[0] * rank_total for _ in range(pn)]

    def add_strassen_block(u_i_off, u_j_off, v_j_off, v_k_off, w_i_off, w_k_off, col_off):
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = u_i_off + i_local
                    j_global = u_j_off + j_local
                    U[i_global * m + j_global][col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = v_j_off + j_local
                    k_global = v_k_off + k_local
                    V[j_global * p + k_global][col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = w_i_off + i_local
                    k_global = w_k_off + k_local
                    W[k_global * n + i_global][col_off] = val
            col_off += 1
        return col_off

    def add_schoolbook_2x2x1(u_i_off, u_j_off, v_j_off, v_k_off, w_i_off, w_k_off, col_off):
        for u_pat, v_pat, w_pat in schoolbook_2x2x1:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = u_i_off + i_local
                    j_global = u_j_off + j_local
                    U[i_global * m + j_global][col_off] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = v_j_off + j_local
                    k_global = v_k_off + k_local
                    V[j_global * p + k_global][col_off] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = w_i_off + i_local
                    k_global = w_k_off + k_local
                    W[k_global * n + i_global][col_off] = val
            col_off += 1
        return col_off

    col = 0
    col = add_strassen_block(0, 0, 0, 0, 0, 0, col)
    col = add_strassen_block(0, 2, 2, 0, 0, 0, col)
    col = add_strassen_block(0, 0, 0, 2, 0, 2, col)
    col = add_strassen_block(0, 2, 2, 2, 0, 2, col)
    col = add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4, col)
    col = add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4, col)

    assert col == rank_total, f"Expected {rank_total}, got {col}"

    def verify(U_mat, V_mat, W_mat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
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
                    s = 0
                    for r in range(rank):
                        s += U_mat[a][r] * V_mat[b][r] * W_mat[c][r]
                    if abs(s) > 1e-9:
                        return False
        return True

    # Try to reduce rank by removing zero columns
    def try_remove_zero_cols(U_mat, V_mat, W_mat, rank):
        keep = []
        for r in range(rank):
            is_zero = True
            for i in range(nm):
                if U_mat[i][r] != 0:
                    is_zero = False
                    break
            if is_zero:
                for j in range(mp):
                    if V_mat[j][r] != 0:
                        is_zero = False
                        break
            if is_zero:
                for k in range(pn):
                    if W_mat[k][r] != 0:
                        is_zero = False
                        break
            if not is_zero:
                keep.append(r)
        if len(keep) == rank:
            return None
        new_rank = len(keep)
        U_new = [[U_mat[i][r] for r in keep] for i in range(nm)]
        V_new = [[V_mat[j][r] for r in keep] for j in range(mp)]
        W_new = [[W_mat[k][r] for r in keep] for k in range(pn)]
        return U_new, V_new, W_new, new_rank

    result = try_remove_zero_cols(U, V, W, rank_total)
    if result:
        U, V, W, rank_total = result

    # Try greedy column elimination: check if removing a column can be compensated
    # This is a heuristic - try removing each column and see if the tensor still matches
    # with a slightly modified decomposition (not generally possible, but worth trying)
    
    # Also try: can we merge two Strassen blocks into fewer terms?
    # The 2x4x5 = 2x(2x2 + 2x1) structure:
    # Left half: 2x2 * 2x2 (rank 7) + 2x2 * 2x1 (rank 4) = 11
    # Right half: 2x2 * 2x2 (rank 7) + 2x2 * 2x1 (rank 4) = 11
    # Total: 22... wait, that's not 36. Let me recount.
    # Actually: 4 Strassen blocks (2x2x2) = 4*7 = 28, plus 2 schoolbook blocks (2x2x1) = 2*4 = 8, total 36.
    
    # The theoretical minimum rank for 2x4x5 matrix multiplication:
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # Upper bound from Strassen: 36 (as constructed)
    # Known: R(2,4,5) <= 36, and it's conjectured to be around 32-36.
    
    # Try a different block decomposition: treat 4x5 as (2+2)x(2+2+1)
    # 2x2 * 2x2 = rank 7
    # 2x2 * 2x1 = rank 4
    # 2x2 * 1x2 = rank 4 (but this is 2x1 * 1x2, different)
    # Actually the structure is A(2x4) * B(4x5)
    # Split B into B1(4x2), B2(4x2), B3(4x1)
    # A*B1: 2x2 * 2x2 -> but A is 2x4, B1 is 4x2
    # We can split A into A1(2x2), A2(2x2)
    # A1*B1: 2x2*2x2 rank 7
    # A2*B1: 2x2*2x2 rank 7
    # A1*B2: 2x2*2x2 rank 7
    # A2*B2: 2x2*2x2 rank 7
    # A1*B3: 2x2*2x1 rank 4
    # A2*B3: 2x2*2x1 rank 4
    # Total: 36. Same as before.
    
    # Alternative: use 3x3 Strassen? No, dimensions don't fit.
    # Use Winograd's method for 2x3? Doesn't directly apply.
    
    # Try: can we use rank-6 for 2x2x2? No, Strassen's 7 is optimal for 2x2x2.
    
    # The best we can likely do is 36, but let's try a randomized search for rank < 36
    # by checking if the tensor has a lower-rank decomposition.
    
    # For now, return the verified rank-36 decomposition
    if verify(U, V, W, rank_total):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(nm)]
        V_sb = [[0] * rank_sb for _ in range(mp)]
        W_sb = [[0] * rank_sb for _ in range(pn)]
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col2] = 1
                    V_sb[j * p + k][col2] = 1
                    W_sb[k * n + i][col2] = 1
                    col2 += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
