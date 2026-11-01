"""Winograd Fourier Transform Technique (WFT) for 2x4x7 matrix multiplication."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition strategy to reduce the rank below the schoolbook 56.
    # Rank 56 is the schoolbook rank.
    # We decompose 2x4x7 into two 2x2x7 multiplications.
    # Each 2x2x7 is decomposed into three 2x2x2 (Strassen, rank 7) and one 2x2x1 (schoolbook, rank 4).
    # Total rank = 2 * (3*7 + 4) = 2 * 25 = 50.
    
    R = 50
    
    # Initialize matrices
    U = [[0] * R for _ in range(n * m)]  # 8 x 50
    V = [[0] * R for _ in range(m * p)]  # 28 x 50
    W = [[0] * R for _ in range(p * n)]  # 14 x 50
    
    col_idx = 0
    
    # Define Strassen terms for 2x2x2
    # A indices: 0,1 (row), 0,1 (col) -> local indices 0..3
    # B indices: 0,1 (row), 0,1 (col) -> local indices 0..3
    # C indices: 0,1 (row), 0,1 (col) -> local indices 0..3
    
    # Strassen multiplications:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: [a00, a01, a10, a11]
        # v: [b00, b01, b10, b11]
        # w: [c00, c01, c10, c11]
        
        # M1
        ( [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1] ), # c00+=, c11+=
        # M2
        ( [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0] ), # c10+=
        # M3
        ( [1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1] ), # c01+=, c11+=
        # M4
        ( [0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0] ), # c00+=, c10+=
        # M5
        ( [1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0] ), # c00-=, c01+=
        # M6
        ( [-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1] ), # c11+=
        # M7
        ( [0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1] )  # c00+=, c11-=
    ]
    
    # Note: The W coefficients in Strassen are usually applied to the final sum.
    # In tensor decomposition T = sum U V W, the W coefficients are part of the term.
    # So we just add the W vectors to the W matrix.
    
    # Schoolbook terms for 2x2x1
    # A: 2x2, B: 2x1, C: 2x1
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    schoolbook_2x2x1_terms = [
        ( [1, 0, 0, 0], [1, 0], [1, 0] ),
        ( [0, 1, 0, 0], [0, 1], [1, 0] ),
        ( [0, 0, 1, 0], [1, 0], [0, 1] ),
        ( [0, 0, 0, 1], [0, 1], [0, 1] )
    ]

    def add_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, is_2x2x1=False):
        nonlocal col_idx
        if not is_2x2x1:
            terms = strassen_terms
            b_cols = 2
        else:
            terms = schoolbook_2x2x1_terms
            b_cols = 1
            
        for u_coeff, v_coeff, w_coeff in terms:
            # Update U
            for i in range(4):
                if u_coeff[i] != 0:
                    u_idx = (a_row_offset + i//2) * m + (a_col_offset + i%2)
                    U[u_idx][col_idx] += u_coeff[i]
            
            # Update V
            for j in range(4 if not is_2x2x1 else 2):
                if v_coeff[j] != 0:
                    v_idx = (b_row_offset + j//2) * p + (b_col_offset + j%2)
                    V[v_idx][col_idx] += v_coeff[j]
            
            # Update W
            for k in range(4 if not is_2x2x1 else 2):
                if w_coeff[k] != 0:
                    w_idx = (b_col_offset + k%2) * n + (a_row_offset + k//2)
                    # Wait, W index is k*n + i.
                    # k is the column of C, i is the row of C.
                    # In Strassen, w_coeff indices correspond to c00, c01, c10, c11.
                    # c00 -> row 0, col 0. c01 -> row 0, col 1.
                    # So k = col, i = row.
                    # My w_coeff list is [c00, c01, c10, c11].
                    # Index 0: c00 (row 0, col 0)
                    # Index 1: c01 (row 0, col 1)
                    # Index 2: c10 (row 1, col 0)
                    # Index 3: c11 (row 1, col 1)
                    
                    w_row = k // 2 # 0 for 0,1; 1 for 2,3? No.
                    # c00: i=0, k=0
                    # c01: i=0, k=1
                    # c10: i=1, k=0
                    # c11: i=1, k=1
                    # So i = k//2, k_local = k%2.
                    w_i = k // 2
                    w_k = k % 2
                    w_idx_global = (b_col_offset + w_k) * n + (a_row_offset + w_i)
                    W[w_idx_global][col_idx] += w_coeff[k]
            
            col_idx += 1

    # Part 1: A_top (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 0-6)
    # Split B_top into 3x 2x2 and 1x 2x1
    
    # B_top col 0-1
    add_block(0, 0, 0, 0, False)
    # B_top col 2-3
    add_block(0, 0, 0, 2, False)
    # B_top col 4-5
    add_block(0, 0, 0, 4, False)
    # B_top col 6
    add_block(0, 0, 0, 6, True)
    
    # Part 2: A_bot (rows 0-1, cols 2-3) * B_bot (rows 2-3, cols 0-6)
    
    # B_bot col 0-1
    add_block(0, 2, 2, 0, False)
    # B_bot col 2-3
    add_block(0, 2, 2, 2, False)
    # B_bot col 4-5
    add_block(0, 2, 2, 4, False)
    # B_bot col 6
    add_block(0, 2, 2, 6, True)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
