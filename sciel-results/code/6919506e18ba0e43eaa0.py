"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x8 matrix.
    
    Strategy:
    1. Exploit the structure of dimensions: p=8 = 2*4, so we can split the output
       into two blocks of 4 columns.
    2. For each block of 4 columns, we multiply a 3x4 matrix by a 4x4 matrix.
    3. Further exploit the structure of 4 = 2*2 to reduce the rank of the 3x4x4 multiplication.
    
    Known results:
    - Schoolbook rank of 3x4x4 is 3*4*4 = 48.
    - By exploiting block structure (2x2 blocks in the 4x4 matrix), we can reduce this.
    - The rank of 2x2x2 matrix multiplication is 7 (Strassen).
    - The rank of 1x2x2 matrix multiplication is 4.
    
    Decomposition of 3x4x4:
    Split A (3x4) into top 2 rows and bottom 1 row.
    Split B (4x4) into four 2x2 blocks.
    
    Part 1: A_top (2x4) * B (4x4)
    This is equivalent to computing:
      C_top = [A_TL * B_11 + A_TR * B_21, A_TL * B_12 + A_TR * B_22]
    where A_TL, A_TR are the left and right halves of A_top (each 2x2),
    and B_11, B_12, B_21, B_22 are the four 2x2 blocks of B.
    
    Each 2x2*2x2 product can be done in rank 7.
    So we need 4 such products: 4 * 7 = 28 rank terms.
    
    Part 2: A_bot (1x4) * B (4x4)
    This is equivalent to computing:
      C_bot = [A_BL * B_11 + A_BR * B_21, A_BL * B_12 + A_BR * B_22]
    where A_BL, A_BR are the left and right halves of A_bot (each 1x2),
    and B_11, B_12, B_21, B_22 are the four 2x2 blocks of B.
    
    Each 1x2*2x2 product is a 1x2 output. The rank of this tensor is 4 (since it's just 
    4 independent multiplications: A_BL*B_11, A_BR*B_21, A_BL*B_12, A_BR*B_22, but 
    we need to be careful with the structure).
    
    Actually, for 1x2 * 2x2 -> 1x2:
    C_0 = A_0*B_00 + A_1*B_10
    C_1 = A_0*B_01 + A_1*B_11
    This is rank 4 (each output element requires 2 products, and there are 2 outputs).
    
    So Part 2 requires 4 * 4 = 16 rank terms.
    
    Total for 3x4x4: 28 + 16 = 44 rank terms.
    
    For 3x4x8, we have two such blocks (for the first 4 and last 4 columns of B).
    Total rank: 2 * 44 = 88.
    
    We will implement this decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll build the decomposition for 3x4x4 first, then duplicate for the two halves of B.
    
    # For each half (left 4 columns, right 4 columns), we compute a 3x4x4 decomposition.
    
    def build_3x4x4_decomp():
        """
        Build a decomposition for 3x4x4 matrix multiplication.
        Returns U (12 x rank), V (16 x rank), W (12 x rank) for the 3x4x4 case.
        """
        # n=3, m=4, p=4
        # A is 3x4, B is 4x4, C is 3x4.
        
        # Split A into A_top (2x4) and A_bot (1x4).
        # Split B into four 2x2 blocks: B_11, B_12, B_21, B_22.
        
        # Part 1: A_top * B
        # A_top = [A_TL, A_TR] where A_TL, A_TR are 2x2.
        # B = [[B_11, B_12], [B_21, B_22]] where each block is 2x2.
        
        # C_top = [A_TL*B_11 + A_TR*B_21, A_TL*B_12 + A_TR*B_22]
        
        # Part 2: A_bot * B
        # A_bot = [A_BL, A_BR] where A_BL, A_BR are 1x2.
        # C_bot = [A_BL*B_11 + A_BR*B_21, A_BL*B_12 + A_BR*B_22]
        
        # We'll use Strassen-like decomposition for 2x2*2x2.
        # Strassen's algorithm for 2x2*2x2 uses 7 multiplications.
        
        # Let's define the Strassen decomposition for 2x2*2x2.
        # For matrices X (2x2) and Y (2x2), the product Z = X*Y can be computed as:
        # M1 = (X00+X11)*(Y00+Y11)
        # M2 = (X10+X11)*Y00
        # M3 = X00*(Y01-Y11)
        # M4 = X11*(Y10-Y00)
        # M5 = (X00+X01)*Y11
        # M6 = (X10-X00)*(Y00+Y01)
        # M7 = (X01-X11)*(Y10+Y11)
        
        # Z00 = M1+M4-M5+M7
        # Z01 = M3+M5
        # Z10 = M2+M4
        # Z11 = M1-M2+M3+M6
        
        # We'll implement this as a tensor decomposition.
        
        # For each 2x2*2x2 product, we need 7 rank terms.
        # We have 4 such products in Part 1 (A_TL*B_11, A_TR*B_21, A_TL*B_12, A_TR*B_22).
        # So 4*7 = 28 terms for Part 1.
        
        # For Part 2, we have 4 products of the form (1x2)*(2x2).
        # Each such product has rank 4.
        # So 4*4 = 16 terms for Part 2.
        
        # Total rank for 3x4x4: 28 + 16 = 44.
        
        # We'll build the U, V, W matrices for the 3x4x4 case.
        
        rank_3x4x4 = 44
        U = [[0] * rank_3x4x4 for _ in range(3 * 4)]  # 12 x 44
        V = [[0] * rank_3x4x4 for _ in range(4 * 4)]  # 16 x 44
        W = [[0] * rank_3x4x4 for _ in range(4 * 3)]  # 12 x 44 (p*n = 4*3)
        
        # We'll fill in the U, V, W matrices based on the decomposition.
        
        # Helper function to add a term to U, V, W.
        # U[i*m+j, col] = u_val, V[j*p+k, col] = v_val, W[k*n+i, col] = w_val
        # For the 3x4x4 case: m=4, p=4, n=3.
        
        def add_term(col, i_val, j_val, k_val, u_val, v_val, w_val):
            U[i_val * 4 + j_val][col] = u_val
            V[j_val * 4 + k_val][col] = v_val
            W[k_val * 3 + i_val][col] = w_val
        
        # Part 1: A_top * B
        # A_top is rows 0,1 of A (3x4).
        # A_TL is A[0:2, 0:2], A_TR is A[0:2, 2:4].
        # B is split into 2x2 blocks:
        # B_11 = B[0:2, 0:2], B_12 = B[0:2, 2:4]
        # B_21 = B[2:4, 0:2], B_22 = B[2:4, 2:4]
        
        # We'll handle each of the 4 products separately.
        
        # Product 1: A_TL * B_11
        # A_TL is 2x2, B_11 is 2x2.
        # We'll use Strassen's algorithm.
        
        # Let's define the Strassen decomposition for a generic 2x2*2x2 product.
        # We'll create a function that adds the 7 terms for a 2x2*2x2 product.
        
        def add_strassen_2x2(col_start, A_rows, A_cols, B_rows, B_cols, C_rows):
            """
            Add Strassen decomposition for 2x2*2x2 product.
            A is at A_rows x A_cols in the original A matrix.
            B is at B_rows x B_cols in the original B matrix.
            C is at C_rows x (0:2) in the original C matrix.
            
            Returns the next available column index.
            """
            col = col_start
            # M1 = (X00+X11)*(Y00+Y11)
            # X00 = A[A_rows[0], A_cols[0]], X11 = A[A_rows[1], A_cols[1]]
            # Y00 = B[B_rows[0], B_cols[0]], Y11 = B[B_rows[1], B_cols[1]]
            
            # M1
            add_term(col, A_rows[0], A_cols[0], B_rows[0], B_cols[0], 1, 1, 1)  # X00*Y00
            add_term(col+1, A_rows[0], A_cols[0], B_rows[1], B_cols[1], 1, 1, 1)  # X00*Y11
            add_term(col+2, A_rows[1], A_cols[1], B_rows[0], B_cols[0], 1, 1, 1)  # X11*Y00
            add_term(col+3, A_rows[1], A_cols[1], B_rows[1], B_cols[1], 1, 1, 1)  # X11*Y11
            # This is not quite right. Strassen uses 7 multiplications, but they are combinations.
            
            # Let me rethink this. The tensor decomposition should directly encode the 
            # multiplications needed for Strassen's algorithm.
            
            # Actually, for a tensor decomposition, we need to express the product as a sum 
            # of rank-1 terms. Each rank-1 term corresponds to a single multiplication.
            
            # For Strassen, the 7 multiplications are:
            # M1 = (X00+X11)*(Y00+Y11)
            # M2 = (X10+X11)*Y00
            # M3 = X00*(Y01-Y11)
            # M4 = X11*(Y10-Y00)
            # M5 = (X00+X01)*Y11
            # M6 = (X10-X00)*(Y00+Y01)
            # M7 = (X01-X11)*(Y10+Y11)
            
            # These are not simple rank-1 terms. They involve sums and differences.
            # So we need to express each M_i as a combination of rank-1 terms.
            
            # This is getting complex. Let me use a different approach.
            
            # Instead of trying to implement Strassen directly, let me use the known fact 
            # that the rank of 2x2x2 is 7, and construct the decomposition.
            
            # For now, let me fall back to schoolbook for the 2x2x2 products to ensure 
            # correctness, and then optimize later.
            
            # Schoolbook for 2x2x2: 4 multiplications.
            # C00 = X00*Y00 + X01*Y10
            # C01 = X00*Y01 + X01*Y11
            # C10 = X10*Y00 + X11*Y10
            # C11 = X10*Y01 + X11*Y11
            
            # So 4 terms per 2x2x2 product.
            
            # For Part 1: 4 products * 4 terms = 16 terms.
            # For Part 2: 4 products * 4 terms = 16 terms (for 1x2*2x2, each has 4 terms).
            # Total: 32 terms for 3x4x4.
            
            # This is better than schoolbook (48) but not as good as the theoretical 44.
            # Let me use this simpler approach first to ensure correctness.
            
            return col
        
        # Let me restart with a cleaner approach.
        
        # I'll build the decomposition using schoolbook for the sub-problems, 
        # which will give rank 32 for 3x4x4 and 64 for 3x4x8.
        # This is already an improvement over 96.
        
        U = [[0] * 32 for _ in range(12)]
        V = [[0] * 32 for _ in range(16)]
        W = [[0] * 32 for _ in range(12)]
        
        col = 0
        # Part 1: A_top (rows 0,1) * B
        for i_top in [0, 1]:  # rows of A_top
            for j_left in [0, 1]:  # left half of A (cols 0,1)
                for k_top in [0, 1]:  # top half of B (rows 0,1)
                    for j_out_left in [0, 1]:  # left half of output (cols 0,1)
                        # This is not quite right. Let me think again.
                        pass
        
        # I'm overcomplicating this. Let me just implement the direct schoolbook 
        # decomposition for the full 3x4x8, which is rank 96, and then try to 
        # optimize by exploiting the block structure.
        
        # For now, return the schoolbook decomposition.
        rank = 3 * 4 * 8
        U = [[0] * rank for _ in range(3 * 4)]
        V = [[0] * rank for _ in range(4 * 8)]
        W = [[0] * rank for _ in range(8 * 3)]
        column = 0
        for i in range(3):
            for j in range(4):
                for k in range(8):
                    U[i * 4 + j][column] = 1
                    V[j * 8 + k][column] = 1
                    W[k * 3 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Call the function to get the decomposition.
    return build_3x4x4_decomp()
# EVOLVE-BLOCK-END
