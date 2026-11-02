import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 115.0

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
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
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
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    # Verify the rank-36 Strassen decomposition
    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        best_result = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = total_rank

    # Try to find a rank-35 decomposition via structured search
    # Strategy: Try to merge two of the 2x2x1 boundary blocks using
    # a clever linear combination that exploits the 5th column structure
    
    if time.time() < deadline - 5.0:
        # Attempt rank-35: Try to find if we can reduce by 1
        # The 2x2x1 blocks use 8 rank total (4+4). Try to find a rank-7
        # decomposition for the two 2x2x1 blocks combined with cross terms.
        
        # Alternative: Try a different partition of the 2x4x5 problem
        # Split as (2x2 + 2x2) * (4x5) but with shared Strassen factors
        
        # Try rank 35 with a modified approach:
        # Use 2x2x2 for the 4x4 core (14 rank) + handle the 5th column
        # The 5th column of B gives 2x1 * 4x1 = 2x4 result, needs 8 rank schoolbook
        # But we can share with Strassen structure
        
        # Let's try: 2x4x4 core with Strassen = 14 + 2x4x1 tail
        # Tail: A(2x4) * B_col5(4x1) -> C_col(2x4), rank 8 schoolbook
        # Total: 14 + 8 = 22... that's not right, need to be more careful
        
        # Actually the standard decomposition:
        # A(2x4) = [A1 A2] where A1,A2 are 2x2
        # B(4x5) = [B1 B2 B3] where B1 is 4x2, B2 is 4x2, B3 is 4x1
        # C = A1*B1 + A2*B2 + A1*B3... no this isn't quite right either
        
        # Let me try a direct rank-35 attempt using optimization
        # Initialize from rank-36 and try to zero out one column
        
        U35 = U.copy()
        V35 = V.copy()
        W35 = W.copy()
        
        # Try to find a linear dependency among the 36 rank-1 terms
        # by checking if we can express one term as a combination of others
        # This is equivalent to checking if the tensor has rank <= 35
        
        # For now, try a few structured modifications
        # Modification 1: Try to merge the two 2x2x1 blocks
        # The 2x2x1 blocks are at columns 14-17 and 28-31
        
        # Try: Can we represent the 5th column multiplication with fewer terms?
        # A(2x4) * B(4x1) where B's 5th column
        # This is a 2x4x1 tensor, which has rank at most 4 (not 8)
        # Wait, that's the key insight!
        
        # The multiplication of a 2x4 matrix by a 4x1 vector is a bilinear map
        # (2x4) x (4x1) -> (2x1), which has tensor rank at most 4
        # (just use the 4 entries of the vector as separate multiplications)
        
        # So instead of using 2x2x1 blocks (4 rank each), we can use
        # a 2x4x1 block which needs only 4 rank total for both rows!
        
        # Let me rebuild with this insight:
        # A(2x4) * B(4x5):
        # Split B into B_left(4x4) and B_right(4x1)
        # A * B_left: use Strassen on 2x4x4 = two 2x2x2 blocks = 14 rank
        # A * B_right: 2x4x1 = 4 rank (direct)
        # Total: 14 + 4 = 18? That seems too low...
        
        # Wait, I need to be more careful. The 2x4x4 multiplication:
        # A(2x4) * B(4x4) -> C(2x4)
        # Split A = [A1 A2] (2x2 each), B = [B1; B2] (2x4 each)
        # C = A1*B1 + A2*B2
        # A1*B1: 2x2x4, can use 2 Strassen 2x2x2 blocks = 14 rank
        # A2*B2: 2x2x4, can use 2 Strassen 2x2x2 blocks = 14 rank
        # Total for 2x4x4: 28 rank? That's worse than before.
        
        # Let me reconsider. The original approach:
        # A(2x4) * B(4x5)
        # Split A into A1(2x2), A2(2x2)
        # Split B into B1(2x5), B2(2x5)  [top 2 rows, bottom 2 rows]
        # C = A1*B1 + A2*B2
        # A1*B1: 2x2x5, use 2x2x2 (7 rank) + 2x2x1 (4 rank) = 11 rank
        # A2*B2: 2x2x5, use 2x2x2 (7 rank) + 2x2x1 (4 rank) = 11 rank
        # Total: 22 rank? But we got 36...
        
        # Oh I see, the Strassen 2x2x2 uses 7 rank but I'm using it for
        # 2x2x2 subproblems. Let me recount:
        # 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1 = 7 + 7 + 4 = 18 rank
        # Two of those: 36 rank. Yes that's correct.
        
        # So the key is: can we do 2x2x5 with fewer than 18 rank?
        # 2x2x5 tensor rank: the 2x2x2 part needs 7, the 2x2x1 part needs 4
        # Total 11? No, 2x2x1 needs rank 4 (it's just 4 independent products)
        # Wait, 2x2x1: A(2x2) * B(2x1) -> C(2x1)
        # This is bilinear map (2x2) x (2x1) -> (2x1)
        # Tensor rank: at most 4 (one for each A entry times each B entry)
        # Can we do it in fewer? The tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
        # For 2x2x1: sum_{i=0,1} sum_{j=0,1} A[i,j] B[j,0] C[i,0]
        # This is 4 terms, and they're independent, so rank is 4.
        
        # So 2x2x5 needs at least 7 + 4 = 11 rank? But we're using 18.
        # The issue is that Strassen 2x2x2 uses 7 rank for 8 products,
        # but we have two 2x2x2 blocks and one 2x2x1 block.
        
        # Hmm, actually for 2x2x5, we can potentially do better by
        # using a combined Strassen-like approach.
        
        # For now, let's stick with the verified rank-36 solution.
        # The search for rank-35 would require deeper analysis.
        
        pass

    # Return the best result found
    return best_result
# EVOLVE-BLOCK-END
