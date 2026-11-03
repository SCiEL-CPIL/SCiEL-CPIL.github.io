import numpy as np
import time
from itertools import product, combinations

def construct():
    n, m, p = 2, 4, 5
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 140.0

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

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        # Check diagonal terms: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = 0
                    for r in range(rank):
                        s += u_val[r] * v_val[r] * w_val[r]
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
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > 1e-9:
                        return False
        return True

    # Try to find a rank-35 decomposition
    # Strategy: Start with rank-36 Strassen decomposition and try to reduce by 1
    # Or try different block structures
    
    # Let's try a 2x4x5 decomposition using:
    # - Two 2x2x5 blocks
    # For 2x2x5, schoolbook rank is 20. Can we do better?
    # Actually, let's think about 2x2xp: the rank is at most 2p (schoolbook)
    # For 2x2x2, Strassen gives rank 7
    # For 2x2x3, we can use: rank 7 (2x2x2) + 2 (2x2x1) = 9, or maybe less
    # For 2x2x5: rank 7 (2x2x2) + 7 (2x2x2) + 2 (2x2x1) = 16? Or 7+7+2=16
    
    # Let me try to construct a 2x2x5 decomposition with rank < 20
    # 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1
    # Rank: 7 + 7 + 2 = 16 (if no sharing)
    # Or: 2x2x5 = 2x2x3 + 2x2x2
    # Rank of 2x2x3: at most 6 (schoolbook), Strassen-like might give 5 or 6
    # Actually for 2x2x3, schoolbook is 6. Can we do 5?
    # Let's assume 2x2x3 has rank 6 for now
    # So 2x2x5: 6 + 7 = 13? Or 7+7+2=16
    
    # For 2x4x5:
    # Option 1: 2x2x5 + 2x2x5 = 16 + 16 = 32 (if we can get 2x2x5 to rank 16)
    # Option 2: 2x3x5 + 2x1x5
    # 2x1x5: rank 5 (schoolbook, can't do better since it's just 5 independent products)
    # 2x3x5: need to find rank
    # 2x3x5 = 2x2x5 + 2x1x5 = 16 + 5 = 21? Or better?
    # Total: 21 + 5 = 26? That seems too optimistic.
    
    # Let me be more careful. The issue is that when we split, the W factors might not be independent.
    
    # Let me try a direct approach: use numerical optimization to find a low-rank decomposition
    # Start from the schoolbook and try to reduce rank
    
    # Actually, let me try the known result: for 2xm x mxp, the rank is at most 2mp - something
    # Known: rank(2x2x2) = 7, rank(2x2x3) <= 9 (or 6 schoolbook)
    # Let me just try to construct explicit decompositions and verify
    
    # Try rank 35 first with a modified Strassen approach
    total_rank = 35
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_strassen(U, V, W, i_off, j_off, k_off, col_start):
        """Add a 2x2x2 Strassen block starting at column col_start"""
        # Strassen patterns for 2x2x2
        # U pattern: [i,j] -> index i*2+j
        # V pattern: [j,k] -> index j*2+k  
        # W pattern: [k,i] -> index k*2+i
        patterns = [
            # (U_vals, V_vals, W_vals) for each of the 7 Strassen terms
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        col = col_start
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = i_off + i_local
                    j_global = j_off + j_local
                    U[i_global * m + j_global, col] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = j_off + j_local
                    k_global = k_off + k_local
                    V[j_global * p + k_global, col] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = i_off + i_local
                    k_global = k_off + k_local
                    W[k_global * n + i_global, col] = val
            col += 1
        return col

    def add_2x2x1_block(U, V, W, i_off, j_off, k_off, col_start):
        """Add a 2x2x1 block (schoolbook, rank 2)"""
        col = col_start
        for i in range(2):
            for j in range(2):
                U[(i_off + i) * m + (j_off + j), col] = 1
                V[(j_off + j) * p + (k_off), col] = 1
                W[(k_off) * n + (i_off + i), col] = 1
                col += 1
        return col

    # Build rank-36 decomposition: 4x (2x2x2) + 2x (2x2x1)
    # 4*7 + 2*2 = 28 + 4 = 32... that's not 36
    # Wait, let me recount: 2x4x5 = 2x2x5 + 2x2x5
    # Each 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1 = 7+7+2 = 16
    # Total: 16 + 16 = 32
    
    # Hmm, that gives 32, not 36. Let me verify the original was 36.
    # Original: 2x2x2 (x4) + 2x2x1 (x2) = 4*7 + 2*2 = 28+4 = 32? 
    # But the original code had total_rank = 36. Let me re-examine.
    
    # Actually looking at the original code more carefully:
    # It has: 2x2x2 at (0,0,0), (0,0,2), (0,2,0), (0,2,2) = 4 blocks of 7 = 28
    # And: 2x2x1 at (0,0,4), (0,2,4) = 2 blocks of 2 = 4
    # Total: 32, but it says 36. There might be an error in the original.
    
    # Let me just try to construct a rank-32 decomposition and verify it
    
    col = 0
    col = add_2x2x2_strassen(U, V, W, 0, 0, 0, col)
    col = add_2x2x2_strassen(U, V, W, 0, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 4, col)
    col = add_2x2x2_strassen(U, V, W, 0, 2, 0, col)
    col = add_2x2x2_strassen(U, V, W, 0, 2, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 4, col)
    
    actual_rank = col
    print(f"Constructed rank: {actual_rank}")
    
    # Truncate to actual rank used
    U = U[:, :actual_rank]
    V = V[:, :actual_rank]
    W = W[:, :actual_rank]
    
    if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), actual_rank):
        print(f"Rank {actual_rank} decomposition verified!")
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(actual_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(actual_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(actual_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        print(f"Rank {actual_rank} decomposition FAILED verification")
    
    # If that didn't work, try the original rank-36 approach
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    col = 0
    col = add_2x2x2_strassen(U, V, W, 0, 0, 0, col)
    col = add_2x2x2_strassen(U, V, W, 0, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 4, col)
    col = add_2x2x2_strassen(U, V, W, 0, 2, 0, col)
    col = add_2x2x2_strassen(U, V, W, 0, 2, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 4, col)
    
    actual_rank = col
    print(f"Second attempt rank: {actual_rank}")
    
    if actual_rank < total_rank:
        U = U[:, :actual_rank]
        V = V[:, :actual_rank]
        W = W[:, :actual_rank]
    
    if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), actual_rank):
        print(f"Rank {actual_rank} decomposition verified (second attempt)!")
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(actual_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(actual_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(actual_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        print(f"Rank {actual_rank} decomposition FAILED (second attempt)")
    
    # Fallback to schoolbook
    return best_result
