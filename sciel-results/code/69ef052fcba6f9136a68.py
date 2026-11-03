<<<<<<< SEARCH
import numpy as np

def construct():
    n, m, p = 4, 4, 7
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 decomposition
    # U, V, W for 2x2x2
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, 0, 0, 0, 0, 1, 1]   # C11
    ]

    # Strassen terms for 4x4x4
    # Term 1: (A00 + A11)(B00 + B11) -> C00, C11
    # Term 2: (A10 + A11)B00 -> C10
    # Term 3: A00(B01 - B11) -> C01
    # Term 4: A11(B10 - B00) -> C10
    # Term 5: (A00 + A01)B11 -> C00, C01
    # Term 6: (A10 - A00)(B00 + B01) -> C11
    # Term 7: (A01 - A11)(B10 + B11) -> C00, C11

    # We need to map the 2x2 indices to 4x4 indices
    # A blocks: 00, 01, 10, 11
    # A indices: 0,1,4,5 (00), 2,3,6,7 (01), 8,9,12,13 (10), 10,11,14,15 (11)
    # B blocks: 00, 01, 10, 11
    # B indices: 0,1,4,5 (00), 2,3,6,7 (01), 8,9,12,13 (10), 10,11,14,15 (11)
    # C blocks: 00, 01, 10, 11
    # C indices: 0,1,8,9 (00), 2,3,10,11 (01), 16,17,24,25 (10), 18,19,26,27 (11)

    def add_term(col_offset, a_blocks, b_blocks, c_blocks, signs_a, signs_b, signs_c):
        # a_blocks: list of 2 indices (0-3) for A
        # b_blocks: list of 2 indices (0-3) for B
        # c_blocks: list of 2 indices (0-3) for C
        # signs: 1 or -1
        for r in range(7):
            col = col_offset + r
            for ur in range(4):
                u_val = U2[ur][r]
                if u_val == 0: continue
                # Map ur to A block
                a_block_idx = a_blocks[ur // 2]
                a_local_idx = ur % 2
                # A block indices
                if a_block_idx == 0:
                    a_global = [0, 1, 4, 5][a_local_idx * 2] if False else [0, 1, 4, 5] # This mapping is tricky
                    # 2x2 block 00: 0,1,4,5. ur=0->0, ur=1->1, ur=2->4, ur=3->5?
                    # No, ur is 0..3.
                    # 0->00, 1->01, 2->10, 3->11 in 2x2.
                    # In 4x4, A00 is rows 0,1 cols 0,1. Indices 0,1,4,5.
                    # A01 is rows 0,1 cols 2,3. Indices 2,3,6,7.
                    # A10 is rows 2,3 cols 0,1. Indices 8,9,12,13.
                    # A11 is rows 2,3 cols 2,3. Indices 10,11,14,15.
                    pass

    # This mapping is too complex to do manually in the loop.
    # Let's just use the schoolbook for the inner 2x2x2?
    # No, rank 49 is key.
    
    # Alternative: Use the known rank-49 decomposition of 4x4x4.
    # We can generate it by taking the Kronecker product of the 2x2x2 Strassen decomposition.
    
    # For now, let's return the schoolbook to be safe, or a simple improvement.
    # Given the time, I will return the schoolbook rank 112.
    # But the prompt asks for a stronger algorithm.
    
    # Let's try to implement the 98 rank solution properly.
    
    # I will hardcode the 49 rank-1 tensors for 4x4x4.
    # This is tedious but possible.
    
    # For the sake of the response, I will provide the schoolbook solution 
    # as it is guaranteed to work, but note that 98 is possible.
    
    # Actually, let's just return the schoolbook.
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
=======
import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # We view the 4x4 matrix multiplication as a tensor product of two 2x2 multiplications.
    # A is 4x4, B is 4x7.
    # We split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
    # We split B into 2x? blocks. Since B is 4x7, we can view it as:
    # B = [[B00, B01, B02], [B10, B11, B12]] where B0* are 2x? blocks.
    # Wait, B is 4 rows, 7 cols.
    # Rows 0,1 correspond to the first "block row" of B.
    # Rows 2,3 correspond to the second "block row" of B.
    # So B0* are 2x7 matrices? No, that's not right for block multiplication.
    
    # Let's use the standard Kronecker product approach for the rank.
    # The rank of matmul(2,2,2) is 7 (Strassen).
    # The rank of matmul(4,4,4) is at most 7*7 = 49.
    # The rank of matmul(4,4,7) can be computed by splitting the 7 columns.
    # We can split 7 into 2+2+2+1? Or 4+3?
    # Actually, we can treat the 4x4x7 problem as:
    # C = A * B
    # Split A into 2x2 blocks (size 2x2).
    # Split B into 2x2 blocks (size 2x2) for the first 4 columns, and 2x1 for the last 3?
    # No, B is 4x7.
    # Let's split B into three vertical blocks: B1 (4x2), B2 (4x2), B3 (4x3)?
    # Or B1 (4x4) and B2 (4x3)?
    
    # Better approach:
    # Use the known rank-49 decomposition for 4x4x4.
    # Then handle the extra 3 columns of B.
    # The 4x4x4 part uses rank 49.
    # The remaining part is 4x4x3.
    # We can compute 4x4x3 using schoolbook (4*4*3 = 48) or better.
    # Total rank 49 + 48 = 97?
    # Or we can mix them.
    
    # Let's construct the 4x4x4 Strassen decomposition first.
    # U, V, W for 2x2x2 (Strassen)
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, 0, 0, 0, 0, 1, 1]   # C11
    ]
    
    # Indices for 4x4x4
    # A is 4x4. Indices 0..15.
    # A[0:2, 0:2] -> 0,1,4,5
    # A[0:2, 2:4] -> 2,3,6,7
    # A[2:4, 0:2] -> 8,9,12,13
    # A[2:4, 2:4] -> 10,11,14,15
    
    # B is 4x4. Indices 0..15.
    # B[0:2, 0:2] -> 0,1,4,5
    # B[0:2, 2:4] -> 2,3,6,7
    # B[2:4, 0:2] -> 8,9,12,13
    # B[2:4, 2:4] -> 10,11,14,15
    
    # C is 4x4. Indices 0..15.
    # C[0:2, 0:2] -> 0,1,8,9
    # C[0:2, 2:4] -> 2,3,10,11
    # C[2:4, 0:2] -> 16,17,24,25
    # C[2:4, 2:4] -> 18,19,26,27
    
    # We want to build U, V, W for 4x4x4 with rank 49.
    # For each pair of Strassen indices (r1, r2) in 0..6:
    # We add a term.
    
    rank_444 = 49
    U_444 = [[0] * rank_444 for _ in range(16)]
    V_444 = [[0] * rank_444 for _ in range(16)]
    W_444 = [[0] * rank_444 for _ in range(16)]
    
    # Mapping from 2x2 block index to 4x4 indices
    # Block 0 (00): rows 0,1 cols 0,1 -> indices 0,1,4,5
    # Block 1 (01): rows 0,1 cols 2,3 -> indices 2,3,6,7
    # Block 2 (10): rows 2,3 cols 0,1 -> indices 8,9,12,13
    # Block 3 (11): rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # For A and B, the index is i*m + j.
    # For C, the index is k*n + i.
    
    # Let's define the indices for each block.
    # A_block_indices:
    # 0: [0, 1, 4, 5]
    # 1: [2, 3, 6, 7]
    # 2: [8, 9, 12, 13]
    # 3: [10, 11, 14, 15]
    
    # B_block_indices:
    # 0: [0, 1, 4, 5]
    # 1: [2, 3, 6, 7]
    # 2: [8, 9, 12, 13]
    # 3: [10, 11, 14, 15]
    
    # C_block_indices:
    # 0: [0, 1, 8, 9]
    # 1: [2, 3, 10, 11]
    # 2: [16, 17, 24, 25]
    # 3: [18, 19, 26, 27]
    
    A_idx = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    B_idx = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    C_idx = {
        0: [0, 1, 8, 9],
        1: [2, 3, 10, 11],
        2: [16, 17, 24, 25],
        3: [18, 19, 26, 27]
    }
    
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            # U part
            for b1 in range(4):
                val1 = U2[b1][r1]
                if val1 == 0: continue
                for b2 in range(4):
                    val2 = U2[b2][r2]
                    if val2 == 0: continue
                    val = val1 * val2
                    # The block index in 4x4 is determined by b1 (row block) and b2 (col block)?
                    # No. U2[b] corresponds to the b-th entry of the 2x2 matrix.
                    # b=0: (0,0), b=1: (0,1), b=2: (1,0), b=3: (1,1).
                    # So b1 determines the row block, b2 determines the col block.
                    # Row block: b1 // 2? No.
                    # b=0 -> row 0, col 0.
                    # b=1 -> row 0, col 1.
                    # b=2 -> row 1, col 0.
                    # b=3 -> row 1, col 1.
                    # So row_block = b // 2, col_block = b % 2.
                    row_block1 = b1 // 2
                    col_block1 = b1 % 2
                    row_block2 = b2 // 2
                    col_block2 = b2 % 2
                    
                    # The 4x4 block index is row_block * 2 + col_block.
                    # 0: (0,0) -> 0
                    # 1: (0,1) -> 1
                    # 2: (1,0) -> 2
                    # 3: (1,1) -> 3
                    block_idx = row_block1 * 2 + col_block1 if False else 0 # Wait.
                    
                    # Actually, the Kronecker product structure is:
                    # A = A1 \otimes A2? No.
                    # A is 4x4. We can view it as A[i,j] where i=i1*2+i2, j=j1*2+j2.
                    # The tensor is T[i,j,k] = sum_l A[i,l] B[l,k].
                    # If we decompose A, B, C into 2x2 blocks, the multiplication is:
                    # C_block[r1,c1] = sum_{r2,c2} A_block[r1,r2] * B_block[r2,c2].
                    # This is exactly the 2x2 matrix multiplication where the entries are 2x2 matrices.
                    # So the rank is the rank of 2x2x2 matmul (which is 7) times the rank of 2x2x2 matmul (which is 7)?
                    # No, the entries are 2x2 matrices. The multiplication of entries is 2x2x2 matmul.
                    # So we need to apply Strassen to the "outer" 2x2 structure, and for each "inner" entry, we need to multiply 2x2 matrices.
                    # This means we need 7 * 7 = 49 terms.
                    
                    # For each outer term (r1, r2) in the Strassen decomposition of the 2x2 block matrix:
                    # We have a product of "blocks".
                    # The outer Strassen term involves linear combinations of blocks.
                    # For each such combination, we multiply the resulting 2x2 matrices using Strassen (or schoolbook).
                    # To get rank 49, we use Strassen for both outer and inner.
                    
                    # So, for a fixed (r1, r2):
                    # The term corresponds to (sum_{b1} U2[b1][r1] * A_block[b1]) * (sum_{b2} V2[b2][r2] * B_block[b2]).
                    # Wait, the indices are mixed.
                    # Let's be precise.
                    # Outer Strassen:
                    # M_r = (sum_{i,j} U2[i,j][r] A_{i,j}) * (sum_{i,j} V2[i,j][r] B_{i,j})
                    # where A_{i,j} is the (i,j)-th 2x2 block of A.
                    # This M_r is a 2x2 matrix (product of two 2x2 matrices).
                    # To compute M_r, we use Strassen on the 2x2 matrices.
                    # Let M_r = S_r \otimes T_r? No.
                    # M_r = P_r * Q_r where P_r, Q_r are 2x2 matrices.
                    # P_r = sum_{i,j} U2[i,j][r] A_{i,j}
                    # Q_r = sum_{i,j} V2[i,j][r] B_{i,j}
                    # Then M_r = P_r Q_r.
                    # We decompose P_r Q_r using Strassen:
                    # M_r = sum_{s=1}^7 W2[i,j][s] * (P_r^{(i)} Q_r^{(j)})? No.
                    # Strassen computes C = AB where A, B are 2x2.
                    # C_{i,j} = sum_s W2[i,j][s] M_s
                    # M_s = (sum_{i,j} U2[i,j][s] A_{i,j}) * (sum_{i,j} V2[i,j][s] B_{i,j})
                    # Here A and B are 2x2 matrices.
                    # So for the inner multiplication, we need to compute M_s for s=1..7.
                    # But A and B here are P_r and Q_r, which are linear combinations of 2x2 blocks.
                    # This is getting nested.
                    
                    # Alternative:
                    # The total tensor is the Kronecker product of the 2x2x2 tensor with itself.
                    # T_4 = T_2 \otimes T_2.
                    # The rank is R(T_2) * R(T_2) = 7 * 7 = 49.
                    # The decomposition is:
                    # For each (r1, r2) in 1..7 x 1..7:
                    # Term = U2 \otimes U2, V2 \otimes V2, W2 \otimes W2.
                    
                    # So, for fixed (r1, r2):
                    # The coefficient for A[i,j] is U2[i1,i2][r1] * U2[j1,j2][r2]?
                    # No.
                    # Let A be 4x4. A[i,j] where i=i1*2+i2, j=j1*2+j2.
                    # The term is:
                    # sum_{i,j} (U2[i1,j1][r1] * U2[i2,j2][r2]) * A[i,j]
                    # This is not quite right.
                    
                    # Let's use the indices directly.
                    # A[i,j] = A[i1,i2, j1,j2]
                    # B[l,k] = B[l1,l2, k1,k2]
                    # C[i,k] = C[i1,i2, k1,k2]
                    
                    # T[i,j,k] = sum_l A[i,l] B[l,k]
                    # = sum_{l1,l2} A[i1,i2,l1,l2] B[l1,l2,k1,k2]
                    
                    # We can group this as:
                    # (sum_{l1} A[i1,?,l1,?] B[?,l2,k1,?]) ...
                    
                    # The Kronecker product decomposition is:
                    # For each (r1, r2):
                    # U[i,j] += U2[i1,j1][r1] * U2[i2,j2][r2]
                    # V[l,k] += V2[l1,k1][r1] * V2[l2,k2][r2]
                    # W[i,k] += W2[i1,k1][r1] * W2[i2,k2][r2]
                    
                    # Let's verify this.
                    # sum_{r1,r2} U2[i1,j1][r1] U2[i2,j2][r2] * V2[l1,k1][r1] V2[l2,k2][r2] * W2[i1,k1][r1] W2[i2,k2][r2]
                    # = (sum_{r1} U2[i1,j1][r1] V2[l1,k1][r1] W2[i1,k1][r1]) * (sum_{r2} U2[i2,j2][r2] V2[l2,k2][r2] W2[i2,k2][r2])
                    # = (sum_{j1} A[i1,j1] B[j1,k1]) * (sum_{j2} A[i2,j2] B[j2,k2])?
                    # No, the indices are mixed.
                    
                    # Actually, the correct grouping is:
                    # A[i1,i2, l1,l2] B[l1,l2, k1,k2]
                    # We want to decompose this into products of (A[i1,?, l1,?] B[?,l2,k1,?]) etc.
                    
                    # Let's stick to the Kronecker product formula:
                    # U[i,j] = U2[i1,j1][r1] * U2[i2,j2][r2]
                    # where i = i1*2 + i2, j = j1*2 + j2.
                    # Wait, the indices in U2 are (row, col) of the 2x2 block.
                    # So U2[a,b][r] corresponds to the (a,b)-th entry of the 2x2 matrix.
                    # In the 4x4 matrix, the entry (i,j) has i1=i//2, i2=i%2, j1=j//2, j2=j%2.
                    # The "block row" is i1, "block col" is j1.
                    # The "inner row" is i2, "inner col" is j2.
                    
                    # So the term is:
                    # U[i,j] += U2[i1, j1][r1] * U2[i2, j2][r2]
                    
                    # Let's test this with a simple case.
                    # If we use this, the rank is 49.
                    
                    # Let's implement this.
                    
                    pass # We will implement the full construction below.
>>>>>>> REPLACE

I will now provide the full implementation of the `construct` function based on the Kronecker product approach.

