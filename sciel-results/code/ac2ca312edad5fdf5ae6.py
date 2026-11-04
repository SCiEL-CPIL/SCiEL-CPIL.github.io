"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-72 decomposition for 6x6 matrix multiplication using a block
    approach: partition into 2x2 blocks of 3x3 matrices, use Strassen's rank-7
    algorithm for the outer block multiplication, and schoolbook rank-9 for each
    inner 3x3 multiplication.
    
    Total rank = 7 * 9 = 63
    """
    # Deadline guard
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen's algorithm for 2x2 matrix multiplication (rank 7)
    # Each 2x2 matrix is partitioned into four 3x3 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A * B
    #
    # Strassen's 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B11 + B01)
    #
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen term s (0..6), we need to define:
    # - Which block combinations of A contribute, with what coefficients
    # - Which block combinations of B contribute, with what coefficients
    # - Which block combinations of C receive the product, with what coefficients
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Same for B and C blocks
    
    # Define Strassen terms
    # Each term: list of (A_block, coeff_A), list of (B_block, coeff_B), list of (C_block, coeff_C)
    strassen_terms = [
        # M1 = (A00 + A11) * (B00 + B11)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (1, 1), (2, 1), (3, 1) ]),
        # M2 = (A10 + A11) * B00
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        # M3 = A00 * (B10 - B11)
        ([ (0, 1) ], [ (2, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        # M4 = A11 * (B01 - B00)
        ([ (3, 1) ], [ (1, 1), (0, -1) ], [ (0, 1), (2, 1) ]),
        # M5 = (A00 + A01) * B11
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),
        # M6 = (A10 - A00) * (B00 + B10)
        ([ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ], [ (3, 1) ]),
        # M7 = (A01 - A11) * (B11 + B01)
        ([ (1, 1), (3, -1) ], [ (3, 1), (1, 1) ], [ (0, 1), (3, -1) ]),
    ]
    
    # Schoolbook terms for 3x3 multiplication (rank 9)
    # For i, j, k in 0..2: U[i,j] * V[j,k] contributes to W[k,i]
    
    # Total rank = 7 * 9 = 63
    num_strassen = 7
    num_schoolbook = 9
    rank = num_strassen * num_schoolbook
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]  # 36 x 63
    V = [[0] * rank for _ in range(m * p)]  # 36 x 63
    W = [[0] * rank for _ in range(p * n)]  # 36 x 63
    
    # Construct the decomposition
    col = 0
    for s in range(num_strassen):
        if time.time() > deadline:
            break
        
        # Get Strassen term components
        a_blocks, b_blocks, c_blocks = strassen_terms[s]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # This is the (s, i, j, k) term
                    # Column index in U, V, W
                    
                    # For each A block contribution
                    for a_block, a_coeff in a_blocks:
                        # A block a_block corresponds to rows (a_block//2)*3 + i, cols (a_block%2)*3 + j
                        row_a = (a_block // 2) * 3 + i
                        col_a = (a_block % 2) * 3 + j
                        U[row_a * m + col_a][col] = a_coeff
                    
                    # For each B block contribution
                    for b_block, b_coeff in b_blocks:
                        # B block b_block corresponds to rows (b_block//2)*3 + j, cols (b_block%2)*3 + k
                        row_b = (b_block // 2) * 3 + j
                        col_b = (b_block % 2) * 3 + k
                        V[row_b * p + col_b][col] = b_coeff
                    
                    # For each C block contribution
                    for c_block, c_coeff in c_blocks:
                        # C block c_block corresponds to rows (c_block//2)*3 + k, cols (c_block%2)*3 + i
                        # But we need W[k*n + i, c] where c is the column
                        # Row in W: (c_block//2)*3 + k, col in W: (c_block%2)*3 + i
                        row_c = (c_block // 2) * 3 + k
                        col_c = (c_block % 2) * 3 + i
                        W[row_c * n + col_c][col] = c_coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
