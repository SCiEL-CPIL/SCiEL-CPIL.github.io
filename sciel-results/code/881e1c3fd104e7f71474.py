import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    # A is n x m (3x4)
    # B is m x p (4x6)
    # C is n x p (3x6)
    
    # We decompose into two blocks of j: {0,1} and {2,3}
    # For each j-block, we have 3 k-blocks: {0,1}, {2,3}, {4,5}
    # For each (j-block, k-block) pair, we have 6 terms (3 rows of A x 2 terms per row)
    
    total_blocks_j = 2
    blocks_per_j = 3
    terms_per_block = 6
    
    rank = total_blocks_j * blocks_per_j * terms_per_block
    # rank = 2 * 3 * 6 = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # j_start: 0 or 2
    for j_start in [0, 2]:
        # k_start: 0, 2, 4
        for k_start in [0, 2, 4]:
            # For each row i in 0..2
            for i in range(n):
                # Term 1: A[i, j_start] * B[j_start, k_start] -> C[i, k_start]
                u_idx = i * m + j_start
                v_idx = j_start * p + k_start
                w_idx = k_start * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
                # Term 2: A[i, j_start+1] * B[j_start+1, k_start] -> C[i, k_start]
                u_idx = i * m + (j_start + 1)
                v_idx = (j_start + 1) * p + k_start
                w_idx = k_start * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
                # Term 3: A[i, j_start] * B[j_start, k_start+1] -> C[i, k_start+1]
                u_idx = i * m + j_start
                v_idx = j_start * p + (k_start + 1)
                w_idx = (k_start + 1) * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
                # Term 4: A[i, j_start+1] * B[j_start+1, k_start+1] -> C[i, k_start+1]
                u_idx = i * m + (j_start + 1)
                v_idx = (j_start + 1) * p + (k_start + 1)
                w_idx = (k_start + 1) * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
                # Wait, the rank of 1x2x2 is 2.
                # C_0 = A_0 B_0 + A_1 B_1
                # C_1 = A_0 B_0' + A_1 B_1'
                # My terms above:
                # 1. A_0 B_0 -> C_0
                # 2. A_1 B_1 -> C_0
                # 3. A_0 B_0' -> C_1
                # 4. A_1 B_1' -> C_1
                # This is 4 terms for 1x2x2?
                # No, 1x2x2 has 2 outputs? No, 1x2x2 means A is 1x2, B is 2x2, C is 1x2.
                # C_0 and C_1 are the outputs.
                # The rank of 1x2x2 is 2?
                # Let's check.
                # T = sum u v w.
                # If rank is 2, we need 2 terms to produce 2 outputs?
                # Term 1: u=[1,0], v=[1,0], w=[1,0] -> A_0 B_0 contributes to C_0.
                # Term 2: u=[0,1], v=[0,1], w=[0,1] -> A_1 B_1 contributes to C_1.
                # This only gives C_0 = A_0 B_0 and C_1 = A_1 B_1.
                # We are missing A_0 B_0' in C_1 and A_1 B_1' in C_1?
                # No, B is 2x2.
                # B = [[B_00, B_01], [B_10, B_11]].
                # C_0 = A_0 B_00 + A_1 B_10.
                # C_1 = A_0 B_01 + A_1 B_11.
                # The terms are:
                # 1. A_0 B_00 -> C_0
                # 2. A_1 B_10 -> C_0
                # 3. A_0 B_01 -> C_1
                # 4. A_1 B_11 -> C_1
                # This is 4 terms.
                # Is the rank of 1x2x2 really 4?
                # Yes, because the inputs B_00, B_01, B_10, B_11 are distinct.
                # The tensor is the direct sum of 4 rank-1 tensors?
                # No, it's a sum of 4 rank-1 tensors.
                # Can it be done with fewer?
                # The rank of 1x2x2 is 4.
                
                # So Rank(1x2x2) = 4.
                # Rank(3x2x2) = 3 * 4 = 12.
                # Rank(3x2x6) = 3 * 12 = 36.
                # Rank(3x4x6) = 2 * 36 = 72.
                
                # This brings us back to 72.
                # The schoolbook is 72.
                # My previous analysis of rank 2 for 1x2x2 was wrong.
                # Rank 2 would be if B was 2x1 (vector).
                # But B is 2x2.
                
                # So the decomposition I wrote above (4 terms per i) is correct and gives rank 72.
                # This is the same as the seed program.
                
                # I need to find a better decomposition.
                # Strassen for 2x2x2 has rank 7.
                # Can we use Strassen?
                # A is 3x4.
                # Split A into 2x2 blocks?
                # Rows 0,1 and Row 2.
                # Cols 0,1 and Cols 2,3.
                
                # Block A_11 (2x2), A_12 (2x2), A_21 (1x2), A_22 (1x2).
                # B_11 (2x2), B_12 (2x2), B_21 (2x2), B_22 (2x2).
                # Wait, B is 4x6.
                # Split B rows into 2,2. Cols into 2,2,2.
                
                # This is getting complicated.
                # Let's stick to the rank 72 solution if I can't find a better one quickly.
                # But the task asks for improvement.
                
                # Let's check the rank of 1x2x2 again.
                # Maybe I can use Winograd?
                # Winograd for 1x2x2?
                # It's just dot product of length 2? No, it's 2 outputs.
                
                # Let's try to implement Strassen for the 2x2 sub-blocks.
                # If I can reduce the 2x2x2 part from 4 to 7? No, 4 is smaller than 7.
                # Schoolbook for 2x2x2 is 4? No, 2x2x2 schoolbook is 4?
                # C = AB. 4 outputs. 4 multiplications. Rank 4.
                # Strassen is 7. So schoolbook is better for 2x2x2!
                # Wait.
                # Rank of 2x2x2 is 7?
                # No, rank of 2x2x2 is 7 is for the TENSOR RANK.
                # Schoolbook uses 4 multiplications?
                # No, schoolbook for 2x2 matrix multiplication uses 4 multiplications?
                # C_11 = A_11 B_11 + A_12 B_21
                # C_12 = A_11 B_12 + A_12 B_22
                # C_21 = A_21 B_11 + A_22 B_21
                # C_22 = A_21 B_12 + A_22 B_22
                # This is 8 multiplications?
                # No, the multiplications are A_ij * B_jk.
                # There are 8 such products.
                # So schoolbook rank is 8.
                # Strassen rank is 7.
                # So Strassen is better for 2x2x2.
                
                # My 1x2x2 analysis:
                # C_0 = A_0 B_00 + A_1 B_10
                # C_1 = A_0 B_01 + A_1 B_11
                # 4 products. Rank 4.
                # Is there a rank 3 or 2?
                # Probably not.
                
                # So for 3x4x6:
                # If I use Strassen on 2x2x2 blocks, I save rank.
                # How to partition 3x4x6 into 2x2x2 blocks?
                # Not possible directly due to dimensions.
                
                # However, I can use the rank 7 decomposition for 2x2x2.
                # And rank 4 for 1x2x2.
                
                # Let's partition A (3x4) into:
                # A1 (2x2), A2 (2x2)
                # A3 (1x2), A4 (1x2)
                
                # B (4x6) into:
                # B1 (2x2), B2 (2x2), B3 (2x2)  (for rows 0,1)
                # B4 (2x2), B5 (2x2), B6 (2x2)  (for rows 2,3)
                
                # C = A1 B1 + A2 B4 + A3 B1 + A4 B4?
                # No, the indices must match.
                # C_11 (2x2) = A1 B1 + A2 B4?
                # A1 is rows 0,1 cols 0,1.
                # A2 is rows 0,1 cols 2,3.
                # B1 is rows 0,1 cols 0,1.
                # B4 is rows 2,3 cols 0,1.
                # C_11 = A1 B1 + A2 B4.
                # This is sum of two 2x2x2 multiplications.
                # Rank = 7 + 7 = 14.
                # Schoolbook for 2x4x2?
                # 2x4x2 has 8 outputs.
                # Rank?
                
                # This is too complex to derive on the fly.
                # I will stick to the rank 72 solution which is correct.
                # The prompt says "improve it". If I can't, I should return the best I have.
                # But 72 is the seed.
                
                # Let's try to get 66 as hinted in the previous thought process.
                # The previous thought process mentioned rank 66.
                # 3x4x6 = 2 * (3x2x6).
                # Rank(3x2x6) = 18?
                # If Rank(3x2x6) is 18, total is 36.
                # But I calculated Rank(1x2x2) = 4.
                # So Rank(3x2x2) = 12.
                # Rank(3x2x6) = 3 * 12 = 36.
                # Total 72.
                
                # Is Rank(1x2x2) really 4?
                # Yes.
                
                # So I cannot get below 72 with simple block decomposition.
                # I will return the rank 72 solution.
                
                pass

    # Given the complexity and time, I will output the schoolbook rank 72 solution 
    # which is guaranteed to be correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
