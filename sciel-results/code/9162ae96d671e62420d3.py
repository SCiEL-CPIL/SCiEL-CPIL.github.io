"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    We use a recursive block approach:
    - Split the 4x4 * 4x6 multiplication into blocks.
    - Use Strassen-like optimizations for the square parts and schoolbook for the rest.
    
    Specifically, we can decompose the 4x4 * 4x6 as:
    - A 4x4 * 4x4 part (first 4 columns of B) and a 4x4 * 4x2 part (last 2 columns of B).
    - For the 4x4 * 4x4 part, use a rank-19 decomposition (from known results for 4x4 matrix multiplication).
    - For the 4x4 * 4x2 part, use a rank-10 decomposition (schoolbook-like but optimized).
    
    However, to keep it simple and correct, let's use the following approach:
    - For the 4x4 * 4x4 part, use Strassen recursively (rank 19 is known, but let's implement Strassen which gives rank 7 for 2x2, and we can extend).
    - Actually, for 4x4 * 4x4, the known minimal rank is 19 (Lader). Let's implement a rank-19 decomposition.
    
    But implementing Lader's rank-19 decomposition is complex. Instead, let's use a simpler but still optimized approach:
    - Use Strassen for the 4x4 * 4x4 part: this would require splitting 4x4 into 2x2 blocks, and using Strassen at each level.
      Strassen for 2x2 has rank 7. For 4x4, if we use Strassen recursively, we get rank 7^2 = 49? No, that's not right.
      Actually, Strassen's algorithm for n x n matrix multiplication has rank O(n^log2(7)). For n=4, it's 7^(2) = 49? No, that's the number of multiplications, but the tensor rank is different.
      
    Let's step back and use a practical approach:
    - For the 4x4 * 4x4 part, we can use a direct sum of Strassen decompositions for the 2x2 blocks.
      If we split A and B (first 4 cols) into 2x2 blocks:
        A = [A11 A12; A21 A22], B = [B11 B12; B21 B22] (each 2x2)
        C = [C11 C12; C21 C22] where Cij = sum_k Aik * Bkj
        Using Strassen for each of the 4 block multiplications? No, that would be 4 * 7 = 28, but we can do better.
        
    Actually, the correct way is to use Strassen's algorithm on the 4x4 matrix multiplication directly.
    Strassen's algorithm for 4x4:
      - Split into 2x2 blocks (4 blocks of 2x2).
      - Use 7 multiplications of 2x2 matrices, each of which can be done with Strassen (7 multiplications of 1x1).
      - Total rank: 7 * 7 = 49? No, that's the number of scalar multiplications, but the tensor rank is the same as the number of rank-1 terms in the decomposition.
      
    Wait, I think I'm confusing myself. Let me use a known result: the tensor rank for 4x4 * 4x4 matrix multiplication is 19 (Lader, 1974). But implementing this is non-trivial.
    
    Alternative approach: use the schoolbook decomposition but with some optimization.
    For 4x4 * 4x6, the schoolbook rank is 4*4*6 = 96.
    
    Let's try a block decomposition:
    - Split A (4x4) into 2x2 blocks: A = [A11 A12; A21 A22]
    - Split B (4x6) into 2x3 blocks: B = [B11 B12 B13; B21 B22 B23] (each 2x3)
    - C (4x6) = [C11 C12 C13; C21 C22 C23] (each 2x3)
    - C11 = A11*B11 + A12*B21, C12 = A11*B12 + A12*B22, etc.
    
    For each 2x2 * 2x3 multiplication, we can use a rank-6 decomposition (schoolbook for 2x2*2x3 is 2*2*3=12, but we can do better).
    Actually, for 2x2 * 2x3, the minimal rank is 6 (I think). Let me verify:
      A = [a b; c d], B = [e f g; h i j]
      C = [ae+ch, af+ci, ag+cj; be+dh, bf+di, bg+dj]
      We can use 6 rank-1 terms? Probably not, since 2x2*2x2 has rank 7 (Strassen), and 2x2*2x3 is larger.
      
    This is getting too complex. Let's use a simpler but still valid approach:
    - Use the schoolbook decomposition for the entire 4x4 * 4x6, which has rank 96.
    - But we know this is not minimal. However, the task says "rank-minimal exact tensor decomposition", so we need to find a lower rank.
    
    Given the complexity, let's use a hybrid approach:
    - For the first 4 columns of B (i.e., 4x4 * 4x4), use a rank-19 decomposition (Lader's).
    - For the last 2 columns of B (i.e., 4x4 * 4x2), use a rank-10 decomposition.
    - Total rank: 19 + 10 = 29.
    
    But I don't have the explicit Lader decomposition. Let's use Strassen for the 4x4*4x4 part.
    Strassen for 4x4: 
      - It requires 7 multiplications of 2x2 matrices.
      - Each 2x2 multiplication can be done with Strassen (7 scalar multiplications).
      - So total rank: 7 * 7 = 49? No, that's not how it works. The tensor rank of the 4x4*4x4 multiplication using Strassen is 7 (at the block level), but each block multiplication is a 2x2*2x2, which has tensor rank 7. So the total tensor rank is 7 * 7 = 49? I think that's correct.
      
    Wait, no. In Strassen's algorithm for 4x4, you compute 7 products of 2x2 matrices. Each of these 2x2*2x2 products is a tensor of rank 7. So the total number of scalar multiplications is 7 * 7 = 49, and the tensor rank is 49.
    
    For the 4x4 * 4x2 part:
      - Split into 2x2 * 2x2 and 2x2 * 2x2? No, 4x2 is 2x2 blocks: [B1; B2] where B1, B2 are 2x2.
      - C = [C1; C2] where C1 = A11*B1 + A12*B2, C2 = A21*B1 + A22*B2.
      - This is two independent 2x2 * 2x2 multiplications (with different A blocks).
      - Each 2x2*2x2 has rank 7, so total rank: 2 * 7 = 14.
      
    Total rank: 49 + 14 = 63.
    
    This is better than 96. Let's implement this.
    
    Steps:
    1. Define the Strassen decomposition for 2x2 * 2x2 (rank 7).
    2. Use it to build the 4x4 * 4x4 decomposition (rank 49).
    3. Use it to build the 4x4 * 4x2 decomposition (rank 14).
    4. Combine them into the full 4x4 * 4x6 decomposition.
    """
    n, m, p = 4, 4, 6
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [a b; c d], B = [e f; g h]
    # C = [a*e + c*g, a*f + c*h; b*e + d*g, b*f + d*h]
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 - M6
    
    # Represent each rank-1 term as (U_col, V_col, W_col)
    # U is for A, V is for B, W is for C
    # For 2x2 * 2x2, U shape (4, 7), V shape (4, 7), W shape (4, 7)
    
    def strassen_2x2():
        """Return U, V, W for 2x2 * 2x2 Strassen decomposition."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        U[0, 0] = 1; U[1, 0] = 1  # a+d
        V[0, 0] = 1; V[1, 0] = 1  # e+h
        W[0, 0] = 1; W[1, 0] = 1  # contributes to C11 and C22
        # Actually, W should be such that W[i] * M1 gives the contribution to C[i]
        # C11 gets +M1, C22 gets +M1
        # So W[0,0] = 1 (for C11), W[3,0] = 1 (for C22)
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1  # c+d
        V[0, 1] = 1  # e
        W[2, 1] = 1  # contributes to C21
        # C21 gets +M2
        # So W[2,1] = 1
        
        # M3 = a(f-h)
        U[0, 2] = 1  # a
        V[1, 2] = 1; V[0, 2] = -1  # f-h
        W[1, 2] = 1  # contributes to C12
        # C12 gets +M3
        # So W[1,2] = 1
        
        # M4 = d(g-e)
        U[3, 3] = 1  # d
        V[0, 3] = -1; V[1, 3] = 1  # g-e (note: g is B[1,0], e is B[0,0])
        # Wait, B = [e f; g h], so B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        # g-e: V[1,3]=1, V[0,3]=-1
        W[0, 3] = -1  # contributes to C11 with -M4
        W[1, 3] = 1   # contributes to C12 with +M4
        # C11 gets -M4, C12 gets +M4
        # So W[0,3] = -1, W[1,3] = 1
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1  # a+b
        V[3, 4] = 1  # h
        W[0, 4] = 1  # contributes to C11
        W[2, 4] = 1  # contributes to C21
        # C11 gets +M5, C21 gets +M5
        # So W[0,4]=1, W[2,4]=1
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1  # c-a
        V[0, 5] = 1; V[1, 5] = 1  # e+f
        W[3, 5] = -1  # contributes to C22 with -M6
        # C22 gets -M6
        # So W[3,5] = -1
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1  # b-d
        V[0, 6] = 1; V[1, 6] = 1  # g+h (B[1,0]+B[1,1])
        W[0, 6] = -1  # contributes to C11 with -M7
        # C11 gets -M7
        # So W[0,6] = -1
        
        return U, V, W
    
    U2, V2, W2 = strassen_2x2()
    
    # Now build 4x4 * 4x4 decomposition using Strassen at the block level
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22] (each 2x2)
    # C = [C11 C12; C21 C22]
    # Strassen's 7 block products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A12(B21-B11)
    # M4 = A21(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 - M6
    
    # Each Mik is a 2x2 * 2x2 product, so we can use the Strassen decomposition for each.
    # Total rank: 7 * 7 = 49
    
    def build_4x4_x_4x4():
        """Build U, V, W for 4x4 * 4x4 using Strassen."""
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # Block indices for A (4x4):
        # A11: rows 0-1, cols 0-1 -> indices 0,1,2,3
        # A12: rows 0-1, cols 2-3 -> indices 4,5,6,7
        # A21: rows 2-3, cols 0-1 -> indices 8,9,10,11
        # A22: rows 2-3, cols 2-3 -> indices 12,13,14,15
        # Similarly for B (4x4) and C (4x6, but here 4x4)
        
        # For each of the 7 Strassen products, add a 2x2*2x2 decomposition
        strassen_products = [
            # (A_indices, B_indices, C_indices, signs)
            # M1 = (A11+A22)(B11+B22)
            {
                'A_blocks': [(0,1,2,3), (12,13,14,15)],  # A11 + A22
                'B_blocks': [(0,1,2,3), (12,13,14,15)],  # B11 + B22
                'C_blocks': [(0,1,2,3), (12,13,14,15)],  # C11 + C22
                'C_signs': [1, 1]
            },
            # M2 = (A21+A22)B11
            {
                'A_blocks': [(8,9,10,11), (12,13,14,15)],  # A21 + A22
                'B_blocks': [(0,1,2,3)],  # B11
                'C_blocks': [(8,9,10,11)],  # C21
                'C_signs': [1]
            },
            # M3 = A12(B21-B11)
            {
                'A_blocks': [(4,5,6,7)],  # A12
                'B_blocks': [(8,9,10,11), (0,1,2,3)],  # B21 - B11
                'C_blocks': [(4,5,6,7)],  # C12
                'C_signs': [1]
            },
            # M4 = A21(B22-B12)
            {
                'A_blocks': [(8,9,10,11)],  # A21
                'B_blocks': [(12,13,14,15), (4,5,6,7)],  # B22 - B12
                'C_blocks': [(0,1,2,3), (4,5,6,7)],  # C11 - C12
                'C_signs': [-1, 1]
            },
            # M5 = (A11+A12)B22
            {
                'A_blocks': [(0,1,2,3), (4,5,6,7)],  # A11 + A12
                'B_blocks': [(12,13,14,15)],  # B22
                'C_blocks': [(0,1,2,3), (8,9,10,11)],  # C11 + C21
                'C_signs': [1, 1]
            },
            # M6 = (A21-A11)(B11+B12)
            {
                'A_blocks': [(8,9,10,11), (0,1,2,3)],  # A21 - A11
                'B_blocks': [(0,1,2,3), (4,5,6,7)],  # B11 + B12
                'C_blocks': [(12,13,14,15)],  # C22
                'C_signs': [-1]
            },
            # M7 = (A12-A22)(B21+B22)
            {
                'A_blocks': [(4,5,6,7), (12,13,14,15)],  # A12 - A22
                'B_blocks': [(8,9,10,11), (12,13,14,15)],  # B21 + B22
                'C_blocks': [(0,1,2,3)],  # C11
                'C_signs': [-1]
            }
        ]
        
        col = 0
        for prod in strassen_products:
            U_prod = np.zeros((4, 7))
            V_prod = np.zeros((4, 7))
            W_prod = np.zeros((4, 7))
            
            # Build U_prod from A_blocks
            for idx, block in enumerate(prod['A_blocks']):
                for local_idx, global_idx in enumerate(block):
                    U_prod[local_idx, :] = U2[local_idx, :]
                # But this is not correct. We need to sum the blocks.
                # Actually, U_prod should be the sum of the U2 for each block, with appropriate signs.
                # Let me restructure this.
            pass
        
        # This approach is getting too complex. Let's use a simpler method:
        # Directly construct the 4x4*4x4 Strassen decomposition by iterating over the 7 products
        # and for each, iterating over the 7 Strassen scalar products.
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Define the 7 Strassen block products more clearly
        # Each block product Mik corresponds to a 2x2*2x2 multiplication
        # For each Mik, we use the 2x2*2x2 Strassen decomposition (U2, V2, W2)
        
        # M1 = (A11+A22)(B11+B22)
        # This is a 2x2*2x2 product where:
        # A_part = A11 + A22 (4x4 -> but we take the 2x2 blocks)
        # Actually, A11 and A22 are 2x2 matrices, so A11+A22 is a 2x2 matrix.
        # Similarly for B.
        
        # Let's define the mapping from global indices to local 2x2 indices
        # For a 2x2 block at position (row_block, col_block) in a 4x4 matrix:
        # global_index = row_block * 4 + col_block * 2 + local_row * 2 + local_col
        # where local_row, local_col in {0,1}
        
        # For A (4x4):
        # A11: row_block=0, col_block=0 -> indices 0,1,2,3
        #   local (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # A12: row_block=0, col_block=1 -> indices 4,5,6,7
        # A21: row_block=1, col_block=0 -> indices 8,9,10,11
        # A22: row_block=1, col_block=1 -> indices 12,13,14,15
        
        # Similarly for B (4x4) and C (4x4)
        
        # For each Strassen product Mik, we have:
        # A_part: a 2x2 matrix formed by summing certain blocks of A
        # B_part: a 2x2 matrix formed by summing certain blocks of B
        # C_part: a 2x2 matrix formed by summing certain blocks of C
        # The product A_part * B_part contributes to C_part with a certain sign
        
        # Let's define the 7 products explicitly with their A, B, C block contributions
        
        # M1: A = A11 + A22, B = B11 + B22, C = C11 + C22 (sign +1 for both)
        # M2: A = A21 + A22, B = B11, C = C21 (sign +1)
        # M3: A = A12, B = B21 - B11, C = C12 (sign +1)
        # M4: A = A21, B = B22 - B12, C = C11 - C12 (sign -1 for C11, +1 for C12)
        # M5: A = A11 + A12, B = B22, C = C11 + C21 (sign +1 for both)
        # M6: A = A21 - A11, B = B11 + B12, C = C22 (sign -1)
        # M7: A = A12 - A22, B = B21 + B22, C = C11 (sign -1)
        
        # For each product, we use the 2x2*2x2 Strassen decomposition (U2, V2, W2)
        # U2 has shape (4,7), V2 (4,7), W2 (4,7)
        # The 4 rows of U2 correspond to the 4 elements of the 2x2 A_part
        # The 4 rows of V2 correspond to the 4 elements of the 2x2 B_part
        # The 4 rows of W2 correspond to the 4 elements of the 2x2 C_part
        
        # We need to map the local 2x2 indices to global 4x4 indices
        
        # For a 2x2 block in A at (row_block, col_block):
        # local (lr, lc) -> global index = row_block * 4 + col_block * 2 + lr * 2 + lc
        # But wait, A is 4x4, so row_block in {0,1}, col_block in {0,1}
        # A11: row_block=0, col_block=0
        # A12: row_block=0, col_block=1
        # A21: row_block=1, col_block=0
        # A22: row_block=1, col_block=1
        
        # For A_part formed by summing blocks, we need to add the U2 contributions with appropriate signs
        
        # Let's define a function to add a 2x2*2x2 Strassen product
        
        def add_strassen_product(U, V, W, col_start, A_blocks, B_blocks, C_blocks, C_signs):
            """
            Add a 2x2*2x2 Strassen product to U, V, W.
            A_blocks: list of (block_row, block_col, sign) for A
            B_blocks: list of (block_row, block_col, sign) for B
            C_blocks: list of (block_row, block_col, sign) for C
            C_signs: not used, the signs are in C_blocks
            """
            for r in range(7):
                # U contribution
                for (br, bc, sign) in A_blocks:
                    for lr in range(2):
                        for lc in range(2):
                            global_idx = br * 4 + bc * 2 + lr * 2 + lc
                            U[global_idx, col_start + r] += sign * U2[lr * 2 + lc, r]
                
                # V contribution
                for (br, bc, sign) in B_blocks:
                    for lr in range(2):
                        for lc in range(2):
                            global_idx = br * 4 + bc * 2 + lr * 2 + lc
                            V[global_idx, col_start + r] += sign * V2[lr * 2 + lc, r]
                
                # W contribution
                for (br, bc, sign) in C_blocks:
                    for lr in range(2):
                        for lc in range(2):
                            global_idx = br * 4 + bc * 2 + lr * 2 + lc
                            W[global_idx, col_start + r] += sign * W2[lr * 2 + lc, r]
        
        # Now define the 7 products
        # Block indices: (row_block, col_block)
        # A11: (0,0), A12: (0,1), A21: (1,0), A22: (1,1)
        # Similarly for B and C
        
        products = [
            # M1
            {
                'A': [(0,0,1), (1,1,1)],  # A11 + A22
                'B': [(0,0,1), (1,1,1)],  # B11 + B22
                'C': [(0,0,1), (1,1,1)]   # C11 + C22
            },
            # M2
            {
                'A': [(1,0,1), (1,1,1)],  # A21 + A22
                'B': [(0,0,1)],            # B11
                'C': [(1,0,1)]             # C21
            },
            # M3
            {
                'A': [(0,1,1)],            # A12
                'B': [(1,0,1), (0,0,-1)],  # B21 - B11
                'C': [(0,1,1)]             # C12
            },
            # M4
            {
                'A': [(1,0,1)],            # A21
                'B': [(1,1,1), (0,1,-1)],  # B22 - B12
                'C': [(0,0,-1), (0,1,1)]   # C11 - C12
            },
            # M5
            {
                'A': [(0,0,1), (0,1,1)],   # A11 + A12
                'B': [(1,1,1)],            # B22
                'C': [(0,0,1), (1,0,1)]    # C11 + C21
            },
            # M6
            {
                'A': [(1,0,1), (0,0,-1)],  # A21 - A11
                'B': [(0,0,1), (0,1,1)],   # B11 + B12
                'C': [(1,1,-1)]            # C22
            },
            # M7
            {
                'A': [(0,1,1), (1,1,-1)],  # A12 - A22
                'B': [(1,0,1), (1,1,1)],   # B21 + B22
                'C': [(0,0,-1)]            # C11
            }
        ]
        
        col = 0
        for prod in products:
            add_strassen_product(U, V, W, col, prod['A'], prod['B'], prod['C'], None)
            col += 7
        
        return U, V, W
    
    U_4x4, V_4x4, W_4x4 = build_4x4_x_4x4()
    
    # Now build 4x4 * 4x2 decomposition
    # B is 4x2, so we can view it as two 2x2 blocks: B1 (top 2 rows, both cols) and B2 (bottom 2 rows, both cols)
    # C is 4x2, so C1 (top 2 rows) and C2 (bottom 2 rows)
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    # This is two independent 2x2*2x2 products (with different A blocks)
    # Each 2x2*2x2 has rank 7, so total rank: 2 * 7 = 14
    
    def build_4x4_x_4x2():
        """Build U, V, W for 4x4 * 4x2."""
        rank = 14
        U = np.zeros((16, rank))
        V = np.zeros((8, rank))  # 4x2 = 8 elements
        W = np.zeros((8, rank))  # 4x2 = 8 elements
        
        # For C1 = A11*B1 + A12*B2:
        # This is a 2x2*2x2 product where:
        # A_part = A11 + A12? No, it's a sum of two products.
        # Actually, C1 = A11*B1 + A12*B2 is not a single 2x2*2x2 product.
        # We need to treat it as the sum of two 2x2*2x2 products: A11*B1 and A12*B2.
        # But that would give rank 7+7=14 for just C1, and another 14 for C2, total 28.
        # That's worse than the schoolbook rank of 4*4*2=32.
        
        # Alternative: use Strassen on the combined 4x2 as a whole?
        # 4x4 * 4x2 can be seen as 4x4 * (4x4 with last 2 cols zero) but that doesn't help.
        
        # Let's use the schoolbook decomposition for 4x4 * 4x2: rank = 4*4*2 = 32.
        # But we can do better by using Strassen for the 2x2 blocks.
        
        # Split A into 2x2 blocks: A11, A12, A21, A22
        # Split B (4x2) into 2x2 blocks: B1 (rows 0-1, cols 0-1), B2 (rows 2-3, cols 0-1)
        # C (4x2) = C1 (rows 0-1, cols 0-1) + C2 (rows 2-3, cols 0-1)
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        
        # For C1: we can use a rank-7 decomposition for A11*B1 and rank-7 for A12*B2, total 14.
        # Similarly for C2: 14.
        # Total: 28.
        
        # But we can use Strassen on the 4x4 * 4x4 where the last 2 columns of B are zero?
        # That would give rank 49, which is worse.
        
        # Let's just use the schoolbook for 4x4 * 4x2: rank 32.
        # Or use a hybrid: for each of the 2 output blocks (C1 and C2), use Strassen for the 2x2*2x2 products.
        # C1 = A11*B1 + A12*B2: this is the sum of two 2x2*2x2 products.
        # We can use Strassen for each: 7 + 7 = 14 for C1.
        # Similarly 14 for C2.
        # Total: 28.
        
        # Let's implement this.
        
        # For C1 = A11*B1 + A12*B2:
        # First product: A11*B1 (2x2*2x2)
        # Second product: A12*B2 (2x2*2x2)
        
        # For A11*B1:
        # A11 is at (0,0) in A, B1 is at (0,0) in B (but B is 4x2, so B1 is rows 0-1, cols 0-1)
        # In the 4x2 B matrix, the indices are:
        # B[0,0], B[0,1] -> row 0
        # B[1,0], B[1,1] -> row 1
        # B[2,0], B[2,1] -> row 2
        # B[3,0], B[3,1] -> row 3
        # So B1 (rows 0-1, cols 0-1) has global indices: 0,1,2,3
        # B2 (rows 2-3, cols 0-1) has global indices: 4,5,6,7
        
        # For C1 (rows 0-1, cols 0-1) in the 4x2 C matrix:
        # C[0,0], C[0,1] -> row 0
        # C[1,0], C[1,1] -> row 1
        # So C1 has global indices: 0,1,2,3
        # C2 has global indices: 4,5,6,7
        
        # Now, for A11*B1:
        # A11 is at (0,0) in A: global indices 0,1,2,3
        # B1 is at (0,0) in B (4x2): global indices 0,1,2,3
        # C1 is at (0,0) in C (4x2): global indices 0,1,2,3
        
        # We can use the 2x2*2x2 Strassen decomposition (U2, V2, W2) for this.
        
        col = 0
        # A11*B1
        for r in range(7):
            for lr in range(2):
                for lc in range(2):
                    # A11 global index
                    global_A = 0 * 4 + 0 * 2 + lr * 2 + lc
                    U[global_A, col + r] += U2[lr * 2 + lc, r]
                    # B1 global index
                    global_B = 0 * 2 + lr * 2 + lc  # B is 4x2, so row * 2 + col
                    V[global_B, col + r] += V2[lr * 2 + lc, r]
                    # C1 global index
                    global_C = 0 * 2 + lr * 2 + lc
                    W[global_C, col + r] += W2[lr * 2 + lc, r]
        col += 7
        
        # A12*B2
        # A12 is at (0,1) in A: global indices 4,5,6,7
        # B2 is at (1,0) in B (4x2): global indices 4,5,6,7
        # C1 is at (0,0) in C: global indices 0,1,2,3
        for r in range(7):
            for lr in range(2):
                for lc in range(2):
                    # A12 global index
                    global_A = 0 * 4 + 1 * 2 + lr * 2 + lc
                    U[global_A, col + r] += U2[lr * 2 + lc, r]
                    # B2 global index
                    global_B = 1 * 2 + lr * 2 + lc
                    V[global_B, col + r] += V2[lr * 2 + lc, r]
                    # C1 global index
                    global_C = 0 * 2 + lr * 2 + lc
                    W[global_C, col + r] += W2[lr * 2 + lc, r]
        col += 7
        
        # Now for C2 = A21*B1 + A22*B2
        # A21*B1
        # A21 is at (1,0) in A: global indices 8,9,10,11
        # B1 is at (0,0) in B: global indices 0,1,2,3
        # C2 is at (1,0) in C: global indices 4,5,6,7
        for r in range(7):
            for lr in range(2):
                for lc in range(2):
                    # A21 global index
                    global_A = 1 * 4 + 0 * 2 + lr * 2 + lc
                    U[global_A, col + r] += U2[lr * 2 + lc, r]
                    # B1 global index
                    global_B = 0 * 2 + lr * 2 + lc
                    V[global_B, col + r] += V2[lr * 2 + lc, r]
                    # C2 global index
                    global_C = 1 * 2 + lr * 2 + lc
                    W[global_C, col + r] += W2[lr * 2 + lc, r]
        col += 7
        
        # A22*B2
        # A22 is at (1,1) in A: global indices 12,13,14,15
        # B2 is at (1,0) in B: global indices 4,5,6,7
        # C2 is at (1,0) in C: global indices 4,5,6,7
        for r in range(7):
            for lr in range(2):
                for lc in range(2):
                    # A22 global index
                    global_A = 1 * 4 + 1 * 2 + lr * 2 + lc
                    U[global_A, col + r] += U2[lr * 2 + lc, r]
                    # B2 global index
                    global_B = 1 * 2 + lr * 2 + lc
                    V[global_B, col + r] += V2[lr * 2 + lc, r]
                    # C2 global index
                    global_C = 1 * 2 + lr * 2 + lc
                    W[global_C, col + r] += W2[lr * 2 + lc, r]
        col += 7
        
        return U, V, W
    
    U_4x2, V_4x2, W_4x2 = build_4x4_x_4x2()
    
    # Now combine the 4x4*4x4 and 4x4*4x2 decompositions
    # The full problem is 4x4 * 4x6
    # B is 4x6, so we split it into B_left (4x4, first 4 cols) and B_right (4x2, last 2 cols)
    # C is 4x6, so C_left (4x4, first 4 cols) and C_right (4x2, last 2 cols)
    # C_left = A * B_left
    # C_right = A * B_right
    
    # U for the full problem: same for both parts (since A is the same)
    # V for the full problem: V_left for B_left (4x4) and V_right for B_right (4x2)
    # W for the full problem: W_left for C_left (4x4) and W_right for C_right (4x2)
    
    # Total rank: 49 + 28 = 77
    
    rank_total = 77
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((24, rank_total))  # 4x6 = 24 elements
    W_full = np.zeros((24, rank_total))  # 4x6 = 24 elements
    
    # Copy U_4x4 to U_full[:, 0:49]
    U_full[:, 0:49] = U_4x4
    
    # Copy U_4x2 to U_full[:, 49:77]
    U_full[:, 49:77] = U_4x2
    
    # Copy V_4x4 to V_full[:, 0:49]
    # V_4x4 is for B_left (4x4), which is the first 4 columns of B_full (4x6)
    # In V_full, the indices for B_full are:
    # B_full[row, col] where row in 0-3, col in 0-5
    # global_index = row * 6 + col
    # B_left is col 0-3, so global_index = row * 6 + col for col 0-3
    for row in range(4):
        for col in range(4):
            global_B_full = row * 6 + col
            # In V_4x4, the index is row * 4 + col
            global_B_left = row * 4 + col
            V_full[global_B_full, 0:49] = V_4x4[global_B_left, 0:49]
    
    # Copy V_4x2 to V_full[:, 49:77]
    # V_4x2 is for B_right (4x2), which is the last 2 columns of B_full (4x6)
    # B_right is col 4-5, so global_index = row * 6 + col for col 4-5
    for row in range(4):
        for col in range(2):
            global_B_full = row * 6 + (4 + col)
            # In V_4x2, the index is row * 2 + col
            global_B_right = row * 2 + col
            V_full[global_B_full, 49:77] = V_4x2[global_B_right, 0:28]
    
    # Copy W_4x4 to W_full[:, 0:49]
    # W_4x4 is for C_left (4x4), which is the first 4 columns of C_full (4x6)
    # In W_full, the indices for C_full are:
    # C_full[row, col] where row in 0-3, col in 0-5
    # global_index = row * 6 + col
    # C_left is col 0-3, so global_index = row * 6 + col for col 0-3
    for row in range(4):
        for col in range(4):
            global_C_full = row * 6 + col
            # In W_4x4, the index is row * 4 + col
            global_C_left = row * 4 + col
            W_full[global_C_full, 0:49] = W_4x4[global_C_left, 0:49]
    
    # Copy W_4x2 to W_full[:, 49:77]
    # W_4x2 is for C_right (4x2), which is the last 2 columns of C_full (4x6)
    # C_right is col 4-5, so global_index = row * 6 + col for col 4-5
    for row in range(4):
        for col in range(2):
            global_C_full = row * 6 + (4 + col)
            # In W_4x2, the index is row * 2 + col
            global_C_right = row * 2 + col
            W_full[global_C_full, 49:77] = W_4x2[global_C_right, 0:28]
    
    # Convert to lists of Fractions
    U_list = [[Fraction(int(x), 1) for x in row] for row in U_full]
    V_list = [[Fraction(int(x), 1) for x in row] for row in V_full]
    W_list = [[Fraction(int(x), 1) for x in row] for row in W_full]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
