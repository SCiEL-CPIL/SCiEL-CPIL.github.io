import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin for 150s limit
    
    n, m, p = 4, 5, 5
    
    # Strategy: Use Strassen for 4x4x4 block (rank 49) and schoolbook for the rest
    # Matrix A is 4x5, B is 5x5, C is 4x5
    # Split A into A_left (4x4) and A_right (4x1)
    # Split B into B_left (5x4) and B_right (5x1)
    # C = A_left * B_left + A_left * B_right + A_right * B_left + A_right * B_right
    # 
    # A_left * B_left: 4x4 * 4x4 -> use Strassen (rank 49)
    # A_left * B_right: 4x4 * 4x1 -> schoolbook (rank 4)
    # A_right * B_left: 4x1 * 5x4 -> schoolbook (rank 4)
    # A_right * B_right: 4x1 * 5x1 -> schoolbook (rank 1)
    # Total: 49 + 4 + 4 + 1 = 58
    
    # Wait, let me reconsider the splitting.
    # A is 4x5: rows 0-3, cols 0-4
    # B is 5x5: rows 0-4, cols 0-4
    # C is 4x5: rows 0-3, cols 0-4
    #
    # Let's split columns of A and rows of B at position 4 (i.e., last column/row)
    # A_left = A[:, 0:4] (4x4), A_right = A[:, 4] (4x1)
    # B_left = B[0:4, :] (4x5), B_right = B[4, :] (1x5)
    #
    # C = A_left @ B_left + A_left @ B_right + A_right @ B_left + A_right @ B_right
    # A_left @ B_left: 4x4 * 4x5 -> this is not square!
    #
    # Hmm, Strassen works for square matrices. Let me think differently.
    #
    # Alternative: Pad to 5x5 or use a different decomposition.
    # Actually, for 4x5 * 5x5, we can think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The schoolbook rank is 4*5*5 = 100.
    #
    # Let me try: split A into two 4x? blocks and B into corresponding blocks.
    # A = [A1 | A2] where A1 is 4x2, A2 is 4x3
    # B = [B1; B2] where B1 is 2x5, B2 is 3x5
    # C = A1*B1 + A2*B2
    # A1*B1: 4x2 * 2x5 -> rank 4*2*5 = 40 schoolbook
    # A2*B2: 4x3 * 3x5 -> rank 4*3*5 = 60 schoolbook
    # Total: 100. No improvement.
    #
    # Better approach: Use the fact that we can apply Strassen to sub-blocks.
    # Let me split A into 2x2 blocks of size 2x2 (since 4=2*2) and handle the 5th column separately.
    #
    # A = [[A00, A01, A02], [A10, A11, A12]] where Aij are 2x2 for j=0,1 and 2x1 for j=2
    # B = [[B00, B01], [B10, B11], [B20, B21]] where Bij are 2x2 for i=0,1 and 1x2 for i=2
    #     Actually B is 5x5, so let's split rows into 2,2,1 and cols into 2,2,1
    # B = [[B00, B01, B02], [B10, B11, B12], [B20, B21, B22]]
    #     where B00,B01,B10,B11 are 2x2, B02,B12 are 2x1, B20,B21 are 1x2, B22 is 1x1
    #
    # C = A*B, C is 4x5
    # Split C into 2x2 blocks:
    # C = [[C00, C01, C02], [C10, C11, C12]]
    # C00 = A00*B00 + A01*B10 + A02*B20
    # C01 = A00*B01 + A01*B11 + A02*B21
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C10 = A10*B00 + A11*B10 + A12*B20
    # C11 = A10*B01 + A11*B11 + A12*B21
    # C12 = A10*B02 + A11*B12 + A12*B22
    #
    # This is getting complex. Let me just use a straightforward approach:
    # Apply Strassen to the 4x4x4 part and schoolbook for the remaining terms.
    #
    # Let A = [A_4x4 | A_4x1] and B = [B_4x5; B_1x5]
    # C = A_4x4 @ B_4x5 + A_4x1 @ B_1x5
    # A_4x4 @ B_4x5: This is 4x4 times 4x5.
    # We can write B_4x5 = [B_4x4 | B_4x1]
    # A_4x4 @ B_4x4: use Strassen (rank 49)
    # A_4x4 @ B_4x1: schoolbook (rank 4*4*1 = 16)
    # A_4x1 @ B_1x5: schoolbook (rank 4*1*5 = 20)
    # Total: 49 + 16 + 20 = 85
    #
    # That's the same as the seed! Let me verify.
    # Seed uses rank 85. My calculation gives 85.
    #
    # Can I do better? 
    # A_4x4 @ B_4x1: 4x4 * 4x1. This is like multiplying a 4x4 matrix by a 4-vector.
    # The rank of this tensor is 4*4*1 = 16 for schoolbook.
    # Can we do better? The matrix-vector product 4x4 * 4x1 has rank 16 in general.
    #
    # A_4x1 @ B_1x5: 4x1 * 1x5. This is an outer product, rank 4*1*5 = 20.
    #
    # So 49 + 16 + 20 = 85. Same as seed.
    #
    # Let me try a different split:
    # A = [A_4x3 | A_4x2] and B = [B_3x5; B_2x5]
    # C = A_4x3 @ B_3x5 + A_4x2 @ B_2x5
    # A_4x3 @ B_3x5: schoolbook rank 4*3*5 = 60
    # A_4x2 @ B_2x5: schoolbook rank 4*2*5 = 40
    # Total: 100. Worse.
    #
    # What if I use Strassen on a 4x4x4 sub-problem and optimize the rest?
    # The best I can get with Strassen on 4x4x4 is 49 + 16 + 20 = 85.
    #
    # Let me try yet another approach: use the fact that 5 = 4 + 1 and apply
    # a recursive Strassen-like decomposition.
    #
    # Actually, let me just implement the 85-rank solution correctly and verify it.
    # The seed already does this. Let me see if I can find a rank < 85.
    #
    # Known bounds: For 4x5*5x5, the rank is at most 85 (as shown above).
    # Can it be lower? The trivial lower bound is max(4*5, 5*5, 4*5) = 25, but that's not tight.
    #
    # Let me try: split into 2x2 blocks where possible.
    # A is 4x5 = 2x(2x5). B is 5x5.
    # Let A = [A_top; A_bot] where A_top is 2x5, A_bot is 2x5.
    # B is 5x5.
    # C_top = A_top @ B (2x5)
    # C_bot = A_bot @ B (2x5)
    #
    # A_top @ B: 2x5 * 5x5. Schoolbook rank 2*5*5 = 50.
    # A_bot @ B: 2x5 * 5x5. Schoolbook rank 2*5*5 = 50.
    # Total: 100.
    #
    # Use Strassen for 2x5*5x5? Not directly applicable.
    #
    # Let me try: A_top @ B where B = [B1 | B2] with B1 5x2, B2 5x3.
    # A_top @ B1: 2x5 * 5x2, rank 2*5*2 = 20
    # A_top @ B2: 2x5 * 5x3, rank 2*5*3 = 30
    # Total for A_top: 50. Same.
    #
    # I think 85 is the best simple decomposition. Let me just implement it correctly.
    
    rank = 85
    U = [[0] * rank for _ in range(n * m)]  # 4*5 = 20 rows
    V = [[0] * rank for _ in range(m * p)]  # 5*5 = 25 rows
    W = [[0] * rank for _ in range(p * n)]  # 5*4 = 20 rows
    
    col = 0
    
    # Part 1: A_4x4 @ B_4x4 using Strassen (rank 49)
    # A_4x4: A[0:4, 0:4], indices in U: i*5+j for i in 0..3, j in 0..3
    # B_4x4: B[0:4, 0:4], indices in V: i*5+j for i in 0..3, j in 0..3
    # C_4x4: C[0:4, 0:4], indices in W: i*4+j for i in 0..3, j in 0..3 (k*n+i where k=col, n=4)
    # Wait, W[k*n+i, r] where k is column index of C (0..4), i is row index (0..3)
    # So W index = k*4 + i
    
    # Strassen for 4x4x4
    # Split 4x4 into 2x2 blocks of 2x2 matrices
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00, B01, B10, B11 similarly
    # C blocks: C00, C01, C10, C11
    
    def a_idx(i, j):
        """Index in U for A[i,j] where i in 0..3, j in 0..3 (for A_4x4 part)"""
        return i * 5 + j
    
    def b_idx(i, j):
        """Index in V for B[i,j] where i in 0..3, j in 0..3 (for B_4x4 part)"""
        return i * 5 + j
    
    def c_idx(k, i):
        """Index in W for C[i,k] where i in 0..3, k in 0..3 (for C_4x4 part)"""
        return k * 4 + i
    
    # 2x2 Strassen terms
    s2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # For 4x4 Strassen, we have 7 products P1..P7, each is a 2x2 matrix product
    # Each Pi is decomposed into 7 rank-1 terms using s2
    # Total: 7 * 7 = 49 terms
    
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # For each Pi, the "matrix" X and Y are 2x2 matrices formed from blocks.
    # We need to map the 2x2 Strassen terms to the 4x4 context.
    
    # Block to global index mapping:
    # A00: a_idx(0,0), a_idx(0,1), a_idx(1,0), a_idx(1,1)
    # A01: a_idx(0,2), a_idx(0,3), a_idx(1,2), a_idx(1,3)
    # A10: a_idx(2,0), a_idx(2,1), a_idx(3,0), a_idx(3,1)
    # A11: a_idx(2,2), a_idx(2,3), a_idx(3,2), a_idx(3,3)
    
    def block_indices(block):
        """Return 4 global indices for a 2x2 block"""
        if block == 0:  # A00/B00/C00
            return [0, 1, 4, 5]  # (0,0), (0,1), (1,0), (1,1)
        if block == 1:  # A01/B01/C01
            return [2, 3, 6, 7]  # (0,2), (0,3), (1,2), (1,3)
        if block == 2:  # A10/B10/C10
            return [8, 9, 12, 13]  # (2,0), (2,1), (3,0), (3,1)
        if block == 3:  # A11/B11/C11
            return [10, 11, 14, 15]  # (2,2), (2,3), (3,2), (3,3)
    
    # For C, the indexing is different: c_idx(k, i) = k*4 + i
    # C00: c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1) = 0, 1, 4, 5
    # C01: c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3) = 2, 3, 6, 7
    # C10: c_idx(2,0), c_idx(2,1), c_idx(3,0), c_idx(3,1) = 8, 9, 12, 13
    # C11: c_idx(2,2), c_idx(2,3), c_idx(3,2), c_idx(3,3) = 10, 11, 14, 15
    
    # So C block indices are the same as A/B block indices! Good.
    
    # Define the 7 products as (A_blocks, B_blocks) with coefficients
    # P1: A = {0:1, 3:1}, B = {0:1, 3:1}
    # P2: A = {2:1, 3:1}, B = {0:1}
    # P3: A = {0:1}, B = {1:1, 3:-1}
    # P4: A = {3:1}, B = {2:1, 0:-1}
    # P5: A = {0:1, 1:1}, B = {3:1}
    # P6: A = {2:1, 0:-1}, B = {0:1, 1:1}
    # P7: A = {1:1, 3:-1}, B = {2:1, 3:1}
    
    products = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1})
    ]
    
    for a_blocks, b_blocks in products:
        # For each of the 7 Strassen terms for 2x2
        for u_local, v_local, w_local in s2:
            # u_local is a vector of 4 coefficients for the 4 elements of the 2x2 matrix X
            # X is formed from A blocks: X = sum c_b * A_b
            # The 4 elements of X are X00, X01, X10, X11
            # X00 = sum over blocks b: c_b * A_b[0,0]
            # etc.
            
            # Build global U vector for this term
            u_global = [0] * 20  # 4*5
            v_global = [0] * 25  # 5*5
            w_global = [0] * 20  # 5*4
            
            # For each element position (0,1,2,3) in the 2x2 matrix
            for pos, coeff in enumerate(u_local):
                if coeff == 0:
                    continue
                # pos 0: X00, pos 1: X01, pos 2: X10, pos 3: X11
                # X00 comes from A_b[0,0] for each block b
                # A_b[0,0] is at global index:
                # Block 0: a_idx(0,0) = 0
                # Block 1: a_idx(0,2) = 2
                # Block 2: a_idx(2,0) = 8
                # Block 3: a_idx(2,2) = 10
                
                # For X00 (pos=0): element [0,0] of each block
                # Block 0: (0,0) -> idx 0
                # Block 1: (0,2) -> idx 2
                # Block 2: (2,0) -> idx 8
                # Block 3: (2,2) -> idx 10
                
                # For X01 (pos=1): element [0,1] of each block
                # Block 0: (0,1) -> idx 1
                # Block 1: (0,3) -> idx 3
                # Block 2: (2,1) -> idx 9
                # Block 3: (2,3) -> idx 11
                
                # For X10 (pos=2): element [1,0] of each block
                # Block 0: (1,0) -> idx 4
                # Block 1: (1,2) -> idx 6
                # Block 2: (3,0) -> idx 12
                # Block 3: (3,2) -> idx 14
                
                # For X11 (pos=3): element [1,1] of each block
                # Block 0: (1,1) -> idx 5
                # Block 1: (1,3) -> idx 7
                # Block 2: (3,1) -> idx 13
                # Block 3: (3,3) -> idx 15
                
                if pos == 0:
                    elem_indices = {0: 0, 1: 2, 2: 8, 3: 10}
                elif pos == 1:
                    elem_indices = {0: 1, 1: 3, 2: 9, 3: 11}
                elif pos == 2:
                    elem_indices = {0: 4, 1: 6, 2: 12, 3: 14}
                else:  # pos == 3
                    elem_indices = {0: 5, 1: 7, 2: 13, 3: 15}
                
                for block_id, block_coeff in a_blocks.items():
                    if block_id in elem_indices:
                        idx = elem_indices[block_id]
                        u_global[idx] += coeff * block_coeff
            
            # Similarly for V
            for pos, coeff in enumerate(v_local):
                if coeff == 0:
                    continue
                if pos == 0:
                    elem_indices = {0: 0, 1: 2, 2: 8, 3: 10}
                elif pos == 1:
                    elem_indices = {0: 1, 1: 3, 2: 9, 3: 11}
                elif pos == 2:
                    elem_indices = {0: 4, 1: 6, 2: 12, 3: 14}
                else:
                    elem_indices = {0: 5, 1: 7, 2: 13, 3: 15}
                
                for block_id, block_coeff in b_blocks.items():
                    if block_id in elem_indices:
                        idx = elem_indices[block_id]
                        v_global[idx] += coeff * block_coeff
            
            # For W, the C block indices are the same
            for pos, coeff in enumerate(w_local):
                if coeff == 0:
                    continue
                if pos == 0:
                    elem_indices = {0: 0, 1: 2, 2: 8, 3: 10}
                elif pos == 1:
                    elem_indices = {0: 1, 1: 3, 2: 9, 3: 11}
                elif pos == 2:
                    elem_indices = {0: 4, 1: 6, 2: 12, 3: 14}
                else:
                    elem_indices = {0: 5, 1: 7, 2: 13, 3: 15}
                
                # W for C blocks: C00, C01, C10, C11
                # But we need to map to the correct C block
                # The w_local coefficients apply to C00, C01, C10, C11 in order
                # So pos 0 -> C00, pos 1 -> C01, pos 2 -> C10, pos 3 -> C11
                
                # For each C block, add the contribution
                # C00: indices 0,1,4,5 in W
                # C01: indices 2,3,6,7
                # C10: indices 8,9,12,13
                # C11: indices 10,11,14,15
                
                c_block_indices = {0: [0,1,4,5], 1: [2,3,6,7], 2: [8,9,12,13], 3: [10,11,14,15]}
                
                # The w_local[pos] coefficient applies to C_block[pos]
                # But we need to distribute to the 4 elements of that C block
                # Actually, w_local is a vector of 4 coefficients for C00, C01, C10, C11
                # Each of these is a scalar, not a 2x2 matrix!
                # Wait, no. In the 2x2 Strassen, w is a vector of 4 coefficients for the 4 elements of the 2x2 result.
                # So w_local[0] is the coefficient for C00[0,0], w_local[1] for C00[0,1], etc.
                # No wait, that's not right either.
                
                # Let me reconsider. In 2x2 Strassen:
                # C = X*Y where X,Y are 2x2.
                # C00 = P1 + P4 - P5 + P7
                # C01 = P3 + P5
                # C10 = P2 + P4
                # C11 = P1 - P2 + P3 + P6
                #
                # Each Pi is a scalar (product of two 2x2 matrices... no, Pi is a 2x2 matrix product).
                # Wait, I'm confusing myself.
                
                # In the standard 2x2 Strassen:
                # A, B are 2x2 matrices.
                # P1 = (A00+A11)(B00+B11) -- this is a scalar product? No!
                # A00, A11, etc. are scalars (elements of the 2x2 matrix).
                # So P1 is a scalar.
                # C00 = P1 + P4 - P5 + P7 (sum of scalars)
                #
                # So in the 2x2 Strassen, u, v, w are vectors of 4 scalars each,
                # representing linear combinations of the 4 elements of A, B, C respectively.
                #
                # Now for 4x4 Strassen:
                # A, B are 4x4 matrices, split into 2x2 blocks of 2x2 matrices.
                # A00, A01, A10, A11 are 2x2 matrices.
                # P1 = (A00+A11)(B00+B11) is a 2x2 matrix product.
                # To decompose P1, we use the 2x2 Strassen terms, but now the "scalars"
                # are replaced by 2x2 matrices.
                #
                # So for P1, we form X = A00+A11 (a 2x2 matrix) and Y = B00+B11 (a 2x2 matrix).
                # Then P1 = X*Y is a 2x2 matrix.
                # We decompose X*Y using 2x2 Strassen, where X and Y are 2x2 matrices.
                # The 2x2 Strassen gives us 7 terms, each with u, v, w vectors of length 4.
                # But now u is a linear combination of the 4 elements of X, which are 2x2 matrices.
                # So u[0] * X00 + u[1] * X01 + u[2] * X10 + u[3] * X11
                # where X00, X01, X10, X11 are 2x2 matrices.
                #
                # This means that for each Strassen term, we need to expand u into
                # contributions to the 4x4 A matrix.
                #
                # My current approach is correct: for each pos (0-3), we identify which
                # elements of which blocks contribute, and add the coefficients.
                #
                # But for W, the situation is different. In the 2x2 Strassen for X*Y,
                # w is a vector of 4 coefficients for the 4 elements of the 2x2 result C_block.
                # So w[0] * C00 + w[1] * C01 + w[2] * C10 + w[3] * C11 where C00 etc. are scalars.
                #
                # But in the 4x4 context, the "result" of P1 is a 2x2 matrix (the C00 block of the 4x4 result).
                # So w[0] applies to C00[0,0], w[1] to C00[0,1], w[2] to C00[1,0], w[3] to C00[1,1].
                #
                # Wait, I think I have it backwards. Let me re-examine.
                #
                # In 2x2 Strassen for matrices X, Y (2x2 each):
                # Result C = X*Y (2x2).
                # C[0,0] = P1 + P4 - P5 + P7
                # C[0,1] = P3 + P5
                # C[1,0] = P2 + P4
                # C[1,1] = P1 - P2 + P3 + P6
                #
                # Each Pi is a scalar (product of two scalars).
                # The u, v, w vectors represent:
                # u: linear combination of X[0,0], X[0,1], X[1,0], X[1,1]
                # v: linear combination of Y[0,0], Y[0,1], Y[1,0], Y[1,1]
                # w: linear combination of C[0,0], C[0,1], C[1,0], C[1,1]
                #
                # So for the 4x4 case, when we apply 2x2 Strassen to compute P1 = X*Y where
                # X = A00+A11 and Y = B00+B11 (both 2x2 matrices):
                # The result is a 2x2 matrix which is the C00 block of the 4x4 result.
                #
                # The w vector from 2x2 Strassen gives coefficients for C[0,0], C[0,1], C[1,0], C[1,1]
                # of the 2x2 result, which correspond to C00[0,0], C00[0,1], C00[1,0], C00[1,1]
                # in the 4x4 result.
                #
                # So my mapping should be:
                # w[0] -> C00[0,0] -> W index 0
                # w[1] -> C00[0,1] -> W index 1
                # w[2] -> C00[1,0] -> W index 4
                # w[3] -> C00[1,1] -> W index 5
                #
                # But wait, that's only for P1 which contributes to C00.
                # What about P2, P3, etc.? They contribute to different C blocks.
                #
                # Actually, in the 4x4 Strassen, the 7 products P1..P7 are 2x2 matrix products.
                # The final 4x4 result C is formed by combining these 2x2 products:
                # C00 = P1 + P4 - P5 + P7 (all 2x2 matrices)
                # C01 = P3 + P5
                # C10 = P2 + P4
                # C11 = P1 - P2 + P3 + P6
                #
                # So each Pi is a 2x2 matrix, and we need to add them to the appropriate C blocks.
                #
                # When I decompose Pi using 2x2 Strassen, I get 7 rank-1 terms for Pi.
                # Each term has u, v, w where w is a vector of 4 coefficients for the 4 elements of Pi.
                #
                # But Pi contributes to multiple C blocks with different coefficients.
                # For example, P1 contributes to C00 (coeff 1), C11 (coeff 1).
                # P2 contributes to C10 (coeff 1), C11 (coeff -1).
                # etc.
                #
                # So I need to track which C blocks each Pi contributes to and with what coefficient.
                #
                # Let me define the C block contributions for each Pi:
                # P1: C00 (+1), C11 (+1)
                # P2: C10 (+1), C11 (-1)
                # P3: C01 (+1), C11 (+1)
                # P4: C00 (+1), C10 (+1)
                # P5: C00 (-1), C01 (+1)
                # P6: C11 (+1)
                # P7: C00 (+1)
                
                # I need to redo this properly. Let me restart the W part.
                pass
            
            # Let me rebuild the W part correctly
            w_global = [0] * 20
            
            # C block contributions for each Pi (index 0-6 corresponds to P1-P7)
            # P1: C00(+1), C11(+1)
            # P2: C10(+1), C11(-1)
            # P3: C01(+1), C11(+1)
            # P4: C00(+1), C10(+1)
            # P5: C00(-1), C01(+1)
            # P6: C11(+1)
            # P7: C00(+1)
            
            pi_idx = products.index((a_blocks, b_blocks))
            
            c_contributions = {
                0: {0: 1, 3: 1},   # P1: C00, C11
                1: {2: 1, 3: -1},  # P2: C10, C11
                2: {1: 1, 3: 1},   # P3: C01, C11
                3: {0: 1, 2: 1},   # P4: C00, C10
                4: {0: -1, 1: 1},  # P5: C00, C01
                5: {3: 1},         # P6: C11
                6: {0: 1}          # P7: C00
            }
            
            for c_block_id, c_coeff in c_contributions[pi_idx].items():
                # c_block_id: 0=C00, 1=C01, 2=C10, 3=C11
                # The 4 elements of this C block in W:
                c_elem_indices = {
                    0: [0, 1, 4, 5],    # C00: W[0], W[1], W[4], W[5]
                    1: [2, 3, 6, 7],    # C01: W[2], W[3], W[6], W[7]
                    2: [8, 9, 12, 13],  # C10: W[8], W[9], W[12], W[13]
                    3: [10, 11, 14, 15] # C11: W[10], W[11], W[14], W[15]
                }
                
                for elem_pos, w_coeff in enumerate(w_local):
                    if w_coeff == 0:
                        continue
                    idx = c_elem_indices[c_block_id][elem_pos]
                    w_global[idx] += w_coeff * c_coeff
            
            # Now add this term to U, V, W
            for i in range(20):
                if u_global[i] != 0:
                    U[i][col] += u_global[i]
            for i in range(25):
                if v_global[i] != 0:
                    V[i][col] += v_global[i]
            for i in range(20):
                if w_global[i] != 0:
                    W[i][col] += w_global[i]
            
            col += 1
    
    # Part 2: A_4x4 @ B_4x1 (schoolbook, rank 16)
    # A_4x4: A[i,j] for i in 0..3, j in 0..3, U index = i*5+j
    # B_4x1: B[i,4] for i in 0..3, V index = i*5+4
    # C_4x1: C[i,4] for i in 0..3, W index = 4*4+i = 16+i
    
    for i in range(4):
        for j in range(4):
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[16+i][col] = 1
            col += 1
    
    # Part 3: A_4x1 @ B_1x5 (schoolbook, rank 20)
    # A_4x1: A[i,4] for i in 0..3, U index = i*5+4
    # B_1x5: B[4,k] for k in 0..4, V index = 4*5+k = 20+k
    # C_4x5: C[i,k] for i in 0..3, k in 0..4, W index = k*4+i
    
    for i in range(4):
        for k in range(5):
            U[i*5+4][col] = 1
            V[20+k][col] = 1
            W[k*4+i][col] = 1
            col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify the decomposition
    if time.time() < deadline:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 if j is the inner index
        # Actually, the tensor is T[i,j,k] = delta(i,j,k) for matrix multiplication
        # T[i,j,k] = 1 if the indices match the multiplication pattern
        # For A (n x m) * B (m x p) = C (n x p):
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (it's the identity tensor for matmul)
        # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Let's verify with random matrices
        A_test = np.random.randn(4, 5)
        B_test = np.random.randn(5, 5)
        C_test = A_test @ B_test
        
        C_decomp = np.zeros((4, 5))
        for r in range(rank):
            U_col = U_np[:, r]
            V_col = V_np[:, r]
            W_col = W_np[:, r]
            
            # Reshape U_col to 4x5, V_col to 5x5, W_col to 5x4
            U_mat = U_col.reshape(4, 5)
            V_mat = V_col.reshape(5, 5)
            W_mat = W_col.reshape(5, 4)
            
            # C += U_mat @ V_mat @ W_mat? No, that's not right.
            # The decomposition is: C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # This is not a simple matrix product.
            
            # Let me compute it directly
            for i in range(4):
                for j in range(5):
                    for k in range(5):
                        C_decomp[i, k] += U_mat[i, j] * V_mat[j, k] * W_mat[k, i]
        
        max_err = np.max(np.abs(C_test - C_decomp))
        if max_err > 1e-10:
            # Fallback to schoolbook
            rank = 100
            U = [[0] * rank for _ in range(20)]
            V = [[0] * rank for _ in range(25)]
            W = [[0] * rank for _ in range(20)]
            col = 0
            for i in range(4):
                for j in range(5):
                    for k in range(5):
                        U[i*5+j][col] = 1
                        V[j*5+k][col] = 1
                        W[k*4+i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
