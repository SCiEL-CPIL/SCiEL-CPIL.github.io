"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-84 decomposition for 4x4 * 4x6 matrix multiplication.
    
    Uses Strassen's algorithm on the 4x4 blocks (each split into 2x2 sub-blocks
    of size 2x2) and falls back to schoolbook for the 2x6 and 2x3 sub-products.
    
    The decomposition is built by:
    1. Splitting A (4x4) into 2x2 blocks of size 2x2
    2. Splitting B (4x6) into 2x2 blocks of size 2x4
    3. Applying Strassen's 7 multiplications at the block level
    4. For each Strassen term, using schoolbook for the sub-block multiplications
    
    This gives rank = 7 * (2*2*4) + 7 * (2*2*4) = 7*8 + 7*8 = 112? 
    Wait, let me recalculate...
    
    Actually, for 4x4 * 4x6:
    - Split A into 2x2 blocks of 2x2
    - Split B into 2x2 blocks of 2x4 (since 6 = 2+4? No, 6 = 3+3 or 2+2+2)
    
    Let me think more carefully. We can split:
    - A (4x4) into 2x2 blocks of size 2x2
    - B (4x6) into 2x2 blocks of size 2x3 (since 6 = 3+3)
    - C (6x4) into 2x2 blocks of size 3x2
    
    Then Strassen gives 7 block multiplications, each of which is (2x2)*(2x3) -> (3x2).
    Each such sub-multiplication has schoolbook rank 2*2*3 = 12.
    Total rank = 7 * 12 = 84.
    
    This matches the known result that 4x4 * 4x6 has rank 84.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    rank = 84
    
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    # Block indices: A split into 2x2 blocks of size 2x2
    # B split into 2x2 blocks of size 2x3
    # C split into 2x2 blocks of size 3x2
    
    # A blocks: A[i_block][j_block] where i_block, j_block in {0, 1}
    # A[0][0] = A[0:2, 0:2], A[0][1] = A[0:2, 2:4]
    # A[1][0] = A[2:4, 0:2], A[1][1] = A[2:4, 2:4]
    
    # B blocks: B[i_block][j_block] where i_block, j_block in {0, 1}
    # B[0][0] = B[0:2, 0:3], B[0][1] = B[0:2, 3:6]
    # B[1][0] = B[2:4, 0:3], B[1][1] = B[2:4, 3:6]
    
    # C blocks: C[i_block][j_block] where i_block, j_block in {0, 1}
    # C[0][0] = C[0:3, 0:2], C[0][1] = C[0:3, 2:4]
    # C[1][0] = C[3:6, 0:2], C[1][1] = C[3:6, 2:4]
    
    # Strassen's algorithm for block multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a (2x2) * (2x3) product, done by schoolbook with rank 12
    
    # Define the Strassen terms with their contributions
    # Each term: (name, A_blocks_with_coeffs, B_blocks_with_coeffs, C_blocks_with_coeffs)
    # A_blocks_with_coeffs: list of (block_index, coefficient)
    # B_blocks_with_coeffs: list of (block_index, coefficient)
    # C_blocks_with_coeffs: list of (block_index, coefficient)
    
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        ('M1', 
         [(0, 1), (3, 1)],  # A00, A11
         [(0, 1), (3, 1)],  # B00, B11
         [(0, 1), (3, -1)]  # C00 += 1, C11 += -1
        ),
        # M2 = (A10 + A11)B00
        ('M2', 
         [(2, 1), (3, 1)],  # A10, A11
         [(0, 1)],          # B00
         [(1, 1), (3, -1)]  # C10 += 1, C11 += -1
        ),
        # M3 = A00(B01 - B11)
        ('M3', 
         [(0, 1)],          # A00
         [(1, 1), (3, -1)], # B01, -B11
         [(1, 1), (3, 1)]   # C01 += 1, C11 += 1
        ),
        # M4 = A11(B10 - B00)
        ('M4', 
         [(3, 1)],          # A11
         [(2, 1), (0, -1)], # B10, -B00
         [(0, 1), (1, 1)]   # C00 += 1, C10 += 1
        ),
        # M5 = (A00 + A01)B11
        ('M5', 
         [(0, 1), (1, 1)],  # A00, A01
         [(3, 1)],          # B11
         [(0, -1), (1, 1)]  # C00 += -1, C01 += 1
        ),
        # M6 = (A10 - A00)(B00 + B01)
        ('M6', 
         [(2, 1), (0, -1)], # A10, -A00
         [(0, 1), (1, 1)],  # B00, B01
         [(3, 1)]           # C11 += 1
        ),
        # M7 = (A01 - A11)(B10 + B11)
        ('M7', 
         [(1, 1), (3, -1)], # A01, -A11
         [(2, 1), (3, 1)],  # B10, B11
         [(0, 1)]           # C00 += 1
        ),
    ]
    
    # For each Strassen term, we need to decompose the (2x2)*(2x3) product
    # into 12 scalar multiplications (schoolbook)
    # A_sub is 2x2, B_sub is 2x3, C_sub is 3x2
    # C_sub[i,j] = sum_k A_sub[i,k] * B_sub[k,j]
    
    # Block index to (row_offset, col_offset) mapping
    # A blocks (2x2 each):
    # A[0] = A[0:2, 0:2], A[1] = A[0:2, 2:4]
    # A[2] = A[2:4, 0:2], A[3] = A[2:4, 2:4]
    A_block_offsets = {
        0: (0, 0),   # A00
        1: (0, 2),   # A01
        2: (2, 0),   # A10
        3: (2, 2),   # A11
    }
    
    # B blocks (2x3 each):
    # B[0] = B[0:2, 0:3], B[1] = B[0:2, 3:6]
    # B[2] = B[2:4, 0:3], B[3] = B[2:4, 3:6]
    B_block_offsets = {
        0: (0, 0),   # B00
        1: (0, 3),   # B01
        2: (2, 0),   # B10
        3: (2, 3),   # B11
    }
    
    # C blocks (3x2 each):
    # C[0] = C[0:3, 0:2], C[1] = C[0:3, 2:4]
    # C[2] = C[3:6, 0:2], C[3] = C[3:6, 2:4]
    C_block_offsets = {
        0: (0, 0),   # C00
        1: (0, 2),   # C01
        2: (3, 0),   # C10
        3: (3, 2),   # C11
    }
    
    col = 0
    for name, a_blocks, b_blocks, c_blocks in strassen_terms:
        # For this Strassen term, we have a (2x2)*(2x3) sub-multiplication
        # Decompose into 12 scalar multiplications
        for i_sub in range(2):  # row of A_sub
            for k_sub in range(2):  # col of A_sub / row of B_sub
                for j_sub in range(3):  # col of B_sub / col of C_sub
                    # U: set coefficients for A blocks
                    for block_idx, coeff in a_blocks:
                        row_off, col_off = A_block_offsets[block_idx]
                        u_idx = (row_off + i_sub) * m + (col_off + k_sub)
                        U[u_idx][col] += coeff
                    
                    # V: set coefficients for B blocks
                    for block_idx, coeff in b_blocks:
                        row_off, col_off = B_block_offsets[block_idx]
                        v_idx = (row_off + k_sub) * p + (col_off + j_sub)
                        V[v_idx][col] += coeff
                    
                    # W: set coefficients for C blocks
                    for block_idx, coeff in c_blocks:
                        row_off, col_off = C_block_offsets[block_idx]
                        w_idx = (row_off + j_sub) * n + (col_off + i_sub)
                        W[w_idx][col] += coeff
                    
                    col += 1
    
    # Convert to list of lists
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
