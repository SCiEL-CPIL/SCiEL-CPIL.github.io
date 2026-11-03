import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def get_strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # P1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1 # C00 += P1, C11 += P1
    
    # P2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1 # C10 += P2
    
    # P3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1 # C01 += P3
    
    # P4 = A11(B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = -1 # C10 += P4, C00 -= P4
    
    # P5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1 # C01 += P5, C11 -= P5
    
    # P6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = -1 # C00 -= P6
    
    # P7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1 # C00 += P7, C11 += P7
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We decompose B into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A B1 + A B2 + A B3
    
    # Get Strassen decomposition for 4x4 multiplication
    # We need to build it from 2x2 Strassen
    
    # Let's build the 4x4 decomposition
    # A is 4x4, B is 4x4
    # Blocks: A00, A01, A10, A11 (each 2x2)
    # B00, B01, B10, B11 (each 2x2)
    
    # Strassen formulas for 4x4:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Each Mi is a 2x2 matrix product, so it has rank 7.
    # We need to expand each Mi into 7 rank-1 terms.
    
    # Total rank for 4x4x4 is 49.
    
    # We will create U, V, W for the full 4x4x9 multiplication.
    # U shape: (16, rank)
    # V shape: (36, rank)
    # W shape: (36, rank)
    
    rank = 49 + 49 + 4 # 102
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Helper to add a 2x2 Strassen decomposition to the global matrices
    # offset_u, offset_v, offset_w are the starting indices in the block matrices
    # scale_u, scale_v, scale_w are multipliers for the entries
    # col_offset is the starting column in U, V, W
    
    def add_strassen_2x2(offset_u, offset_v, offset_w, col_offset, scale_u, scale_v, scale_w):
        # Get the base 2x2 Strassen matrices
        U_base, V_base, W_base = get_strassen_2x2()
        for i in range(4):
            for c in range(7):
                if U_base[i][c] != 0:
                    # Map local index i to global index
                    # i is 0..3, representing (r, c) in 2x2
                    # offset_u is (row_block, col_block) in 4x4? No.
                    # offset_u is the starting index in the 4x4 matrix.
                    # If offset_u = 0, it's A00.
                    # If offset_u = 2, it's A01.
                    # If offset_u = 8, it's A10.
                    # If offset_u = 10, it's A11.
                    
                    # The local index i in U_base corresponds to (i//2, i%2) in the 2x2 block.
                    # So global index is offset_u + i.
                    U[offset_u + i][col_offset + c] += U_base[i][c] * scale_u
                if V_base[i][c] != 0:
                    V[offset_v + i][col_offset + c] += V_base[i][c] * scale_v
                if W_base[i][c] != 0:
                    W[offset_w + i][col_offset + c] += W_base[i][c] * scale_w

    # We need to define the block offsets for A and B.
    # A is 4x4.
    # A00: rows 0-1, cols 0-1. Offset 0.
    # A01: rows 0-1, cols 2-3. Offset 2.
    # A10: rows 2-3, cols 0-1. Offset 8.
    # A11: rows 2-3, cols 2-3. Offset 10.
    
    # B is 4x4.
    # B00: rows 0-1, cols 0-1. Offset 0.
    # B01: rows 0-1, cols 2-3. Offset 2.
    # B10: rows 2-3, cols 0-1. Offset 8.
    # B11: rows 2-3, cols 2-3. Offset 10.
    
    # For the first 4x4 multiplication (A B1):
    # B1 is the first 4 columns of B.
    # B1_00: rows 0-1, cols 0-1. Offset 0.
    # B1_01: rows 0-1, cols 2-3. Offset 2.
    # B1_10: rows 2-3, cols 0-1. Offset 8.
    # B1_11: rows 2-3, cols 2-3. Offset 10.
    
    # For the second 4x4 multiplication (A B2):
    # B2 is the next 4 columns of B (cols 4-7).
    # B2_00: rows 0-1, cols 4-5. Offset 4.
    # B2_01: rows 0-1, cols 6-7. Offset 6.
    # B2_10: rows 2-3, cols 4-5. Offset 12.
    # B2_11: rows 2-3, cols 6-7. Offset 14.
    
    # For the third multiplication (A B3):
    # B3 is the last column of B (col 8).
    # This is 4x1. We use schoolbook.
    
    # Let's implement the 4x4 Strassen decomposition.
    
    # M1 = (A00 + A11)(B00 + B11)
    # This is not a single 2x2 product. It's a product of sums.
    # We need to expand (A00 + A11)(B00 + B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This is 4 separate 2x2 products.
    # So M1 has rank 4 * 7 = 28.
    # Similarly, M2 = (A10 + A11)B00 = A10B00 + A11B00. Rank 14.
    # M3 = A00(B01 - B11) = A00B01 - A00B11. Rank 14.
    # M4 = A11(B10 - B00) = A11B10 - A11B00. Rank 14.
    # M5 = (A00 + A01)B11 = A00B11 + A01B11. Rank 14.
    # M6 = (A10 - A00)(B00 + B01) = A10B00 + A10B01 - A00B00 - A00B01. Rank 28.
    # M7 = (A01 - A11)(B10 + B11) = A01B10 + A01B11 - A11B10 - A11B11. Rank 28.
    
    # Total rank: 28 + 14 + 14 + 14 + 14 + 28 + 28 = 140?
    # No, this is not efficient. The Strassen algorithm reduces the number of multiplications, but if we expand the sums, we lose the benefit.
    # The Strassen algorithm works because we compute the sums first, then multiply.
    # But in a tensor decomposition, we can't "compute sums first". We just have linear combinations.
    # So (A00 + A11)(B00 + B11) is indeed 4 separate products in a rank-1 decomposition.
    # So the rank of 4x4x4 using Strassen is NOT 49.
    # The rank of 4x4x4 is at most 49 if we use a different decomposition, but the standard Strassen recursion gives rank 49 only if we treat the 2x2 blocks as scalars.
    # But here the 2x2 blocks are matrices, so we have to multiply them.
    # So the rank is 7 * 7 = 49? No.
    # If we use Strassen to multiply 4x4 matrices, we view them as 2x2 block matrices.
    # The 7 products are of 2x2 matrices.
    # Each 2x2 matrix multiplication has rank 7.
    # So the total rank is 7 * 7 = 49.
    # This is correct. The linearity of the tensor decomposition allows us to combine the terms.
    # (A00 + A11)(B00 + B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # In the tensor decomposition, this corresponds to 4 separate rank-1 terms? No.
    # The tensor decomposition is linear.
    # So if we have a term $U V W$ where $U = A00 + A11$, $V = B00 + B11$, then it's a single term in the decomposition.
    # But $U$ and $V$ are vectors in the decomposition.
    # So yes, $(A00 + A11)(B00 + B11)$ is a single term in the rank-1 decomposition of the block multiplication.
    # So the rank is indeed 49.
    
    # So we can just use the 49 terms.
    
    # Let's re-verify the offsets.
    # For M1 = (A00 + A11)(B00 + B11):
    # U part: A00 + A11.
    # A00 is at offset 0. A11 is at offset 10.
    # So U has 1s at offsets 0 and 10.
    # V part: B00 + B11.
    # B00 is at offset 0. B11 is at offset 10.
    # So V has 1s at offsets 0 and 10.
    # W part: C00 + C11.
    # C00 is at offset 0. C11 is at offset 10.
    # So W has 1s at offsets 0 and 10.
    
    # This is a single rank-1 term for the block multiplication.
    # But wait, the block multiplication is itself a 2x2 matrix multiplication.
    # So we need to expand this block multiplication into 7 rank-1 terms.
    # So M1 contributes 7 terms to the total rank.
    
    # So the total rank is 7 * 7 = 49.
    
    # Let's implement this.
    
    # We will define the 7 Strassen products for the 4x4 case.
    # Each product is a 2x2 matrix multiplication.
    # We will call add_strassen_2x2 for each product.
    
    # But add_strassen_2x2 expects a single 2x2 block.
    # Here, the "2x2 block" is a sum of blocks.
    # So we need a more general function.
    
    # Let's just hardcode the 49 terms.
    # It's safer and faster.
    
    # I will write a script to generate the 49 terms and paste them here.
    # Or I can write a function that generates them.
    
    # Given the time, I will use the rank 144 schoolbook algorithm as a fallback, and try to implement the 102 rank algorithm.
    # If the 102 rank algorithm is too complex, I will stick to 144.
    
    # Let's try to implement the 102 rank algorithm.
    
    # I will create a function `construct_4x4x4()` that returns U, V, W for 4x4x4 multiplication.
    # Then I will use it to build the 4x4x9 decomposition.
    
    def construct_4x4x4():
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # Strassen formulas
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # We need to expand each Mi into 7 terms.
        # This is complex. I will use the schoolbook algorithm for 4x4x4 (rank 16) and 4x4x1 (rank 4).
        # Total rank 16 + 16 + 4 = 36? No, 4x4x4 is rank 16.
        # So 4x4x9 is rank 16 + 16 + 4 = 36? No, 4x4x8 is rank 32.
        # So 4x4x9 is rank 32 + 4 = 36.
        # This is much better than 144.
        
        # Let's use schoolbook for 4x4x8 and 4x4x1.
        
        rank = 36
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(36)]
        W = [[0]*rank for _ in range(36)]
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    U[i*4+j][col] = 1
                    V[j*9+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        # Last column
        for i in range(4):
            for j in range(4):
                U[i*4+j][col] = 1
                V[j*9+8][col] = 1
                W[8*4+i][col] = 1
                col += 1
                
        return U, V, W
        
    U, V, W = construct_4x4x4()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
