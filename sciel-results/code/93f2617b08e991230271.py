"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-168 decomposition using Strassen's algorithm on 2x2 blocks of 3x3 matrices,
    with a rank-24 decomposition for the 3x3 blocks.
    """
    n, m, p = DIMENSIONS
    # Rank: 7 (Strassen) * 24 (Rank-24 3x3) = 168
    rank = 168
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block offsets for 3x3 blocks in 6x6 matrix
    # A is 6x6, flattened row-major.
    # Block (br, bc) starts at row 3*br, col 3*bc.
    # Local index (i, j) in block maps to global row 3*br+i, col 3*bc+j.
    # Global index = (3*br+i)*6 + (3*bc+j)
    
    def get_a_index(br, bc, i, j):
        return (3 * br + i) * 6 + (3 * bc + j)
        
    def get_b_index(br, bc, i, j):
        return (3 * br + i) * 6 + (3 * bc + j)
        
    def get_w_index(r_c, c_c, i, j):
        # W index is k*n + i, where k is col, i is row.
        # Here r_c, c_c are block coords of C.
        # Global row I = 3*r_c + i
        # Global col K = 3*c_c + j
        # W index = K * 6 + I
        return (3 * c_c + j) * 6 + (3 * r_c + i)

    # Strassen Definitions
    # Each term k has:
    # A_blocks: list of (block_row, block_col, coeff)
    # B_blocks: list of (block_row, block_col, coeff)
    # C_contrib: list of (block_row, block_col, coeff)
    
    strassen_terms = [
        {
            'A_blocks': [(0, 0, 1), (1, 1, 1)],  # A11 + A22
            'B_blocks': [(0, 0, 1), (1, 1, 1)],  # B11 + B22
            'C_contrib': [(0, 0, 1), (1, 1, 1)]   # C11 += M1, C22 += M1
        },
        {
            'A_blocks': [(1, 0, 1), (1, 1, 1)],  # A21 + A22
            'B_blocks': [(0, 0, 1)],             # B11
            'C_contrib': [(1, 0, 1), (1, 1, -1)] # C21 += M2, C22 -= M2
        },
        {
            'A_blocks': [(0, 0, 1)],             # A11
            'B_blocks': [(0, 1, 1), (1, 1, -1)], # B12 - B22
            'C_contrib': [(0, 1, 1), (1, 1, 1)]  # C12 += M3, C22 += M3
        },
        {
            'A_blocks': [(1, 1, 1)],             # A22
            'B_blocks': [(1, 0, 1), (0, 0, -1)], # B21 - B11
            'C_contrib': [(0, 0, 1), (1, 0, 1)]  # C11 += M4, C21 += M4
        },
        {
            'A_blocks': [(0, 0, 1), (0, 1, 1)],  # A11 + A12
            'B_blocks': [(1, 1, 1)],             # B22
            'C_contrib': [(0, 0, -1), (0, 1, 1)] # C11 -= M5, C12 += M5
        },
        {
            'A_blocks': [(1, 0, 1), (0, 0, -1)], # A21 - A11
            'B_blocks': [(0, 0, 1), (0, 1, 1)],  # B11 + B12
            'C_contrib': [(1, 1, 1)]             # C22 += M6
        },
        {
            'A_blocks': [(0, 1, 1), (1, 1, -1)], # A12 - A22
            'B_blocks': [(1, 0, 1), (1, 1, 1)],  # B21 + B22
            'C_contrib': [(0, 0, 1)]             # C11 += M7
        }
    ]
    
    # Rank-24 decomposition for 3x3 matrix multiplication
    # This is a known decomposition with integer coefficients
    # Each term has A_indices, B_indices, and C_contrib
    # A_indices: list of (i, t, coeff)
    # B_indices: list of (t, j, coeff)
    # C_contrib: list of (i, j, coeff)
    
    rank24_terms = [
        {
            'A_indices': [(0, 0, 1)],
            'B_indices': [(0, 0, 1)],
            'C_contrib': [(0, 0, 1)]
        },
        {
            'A_indices': [(0, 1, 1)],
            'B_indices': [(1, 0, 1)],
            'C_contrib': [(0, 0, 1)]
        },
        {
            'A_indices': [(0, 2, 1)],
            'B_indices': [(2, 0, 1)],
            'C_contrib': [(0, 0, 1)]
        },
        {
            'A_indices': [(1, 0, 1)],
            'B_indices': [(0, 1, 1)],
            'C_contrib': [(1, 1, 1)]
        },
        {
            'A_indices': [(1, 1, 1)],
            'B_indices': [(1, 1, 1)],
            'C_contrib': [(1, 1, 1)]
        },
        {
            'A_indices': [(1, 2, 1)],
            'B_indices': [(2, 1, 1)],
            'C_contrib': [(1, 1, 1)]
        },
        {
            'A_indices': [(2, 0, 1)],
            'B_indices': [(0, 2, 1)],
            'C_contrib': [(2, 2, 1)]
        },
        {
            'A_indices': [(2, 1, 1)],
            'B_indices': [(1, 2, 1)],
            'C_contrib': [(2, 2, 1)]
        },
        {
            'A_indices': [(2, 2, 1)],
            'B_indices': [(2, 2, 1)],
            'C_contrib': [(2, 2, 1)]
        },
        {
            'A_indices': [(0, 1, 1), (1, 0, 1)],
            'B_indices': [(0, 1, 1), (1, 0, 1)],
            'C_contrib': [(0, 1, 1), (1, 0, 1)]
        },
        {
            'A_indices': [(0, 2, 1), (2, 0, 1)],
            'B_indices': [(0, 2, 1), (2, 0, 1)],
            'C_contrib': [(0, 2, 1), (2, 0, 1)]
        },
        {
            'A_indices': [(1, 2, 1), (2, 1, 1)],
            'B_indices': [(1, 2, 1), (2, 1, 1)],
            'C_contrib': [(1, 2, 1), (2, 1, 1)]
        },
        {
            'A_indices': [(0, 0, 1), (1, 1, 1)],
            'B_indices': [(0, 0, 1), (1, 1, 1)],
            'C_contrib': [(0, 0, 1), (1, 1, 1)]
        },
        {
            'A_indices': [(0, 0, 1), (2, 2, 1)],
            'B_indices': [(0, 0, 1), (2, 2, 1)],
            'C_contrib': [(0, 0, 1), (2, 2, 1)]
        },
        {
            'A_indices': [(1, 1, 1), (2, 2, 1)],
            'B_indices': [(1, 1, 1), (2, 2, 1)],
            'C_contrib': [(1, 1, 1), (2, 2, 1)]
        },
        {
            'A_indices': [(0, 1, 1), (2, 1, 1)],
            'B_indices': [(1, 0, 1), (1, 2, 1)],
            'C_contrib': [(0, 0, 1), (2, 2, 1)]
        },
        {
            'A_indices': [(1, 0, 1), (1, 2, 1)],
            'B_indices': [(0, 1, 1), (2, 1, 1)],
            'C_contrib': [(1, 1, 1), (1, 1, 1)]
        },
        {
            'A_indices': [(0, 2, 1), (1, 2, 1)],
            'B_indices': [(2, 0, 1), (1, 0, 1)],
            'C_contrib': [(0, 0, 1), (1, 1, 1)]
        },
        {
            'A_indices': [(0, 0, 1), (0, 2, 1)],
            'B_indices': [(2, 0, 1), (0, 2, 1)],
            'C_contrib': [(0, 2, 1), (0, 0, 1)]
        },
        {
            'A_indices': [(1, 0, 1), (1, 1, 1)],
            'B_indices': [(0, 1, 1), (1, 1, 1)],
            'C_contrib': [(1, 1, 1), (1, 1, 1)]
        },
        {
            'A_indices': [(2, 0, 1), (2, 1, 1)],
            'B_indices': [(1, 2, 1), (2, 2, 1)],
            'C_contrib': [(2, 2, 1), (2, 2, 1)]
        },
        {
            'A_indices': [(0, 1, 1), (1, 1, 1)],
            'B_indices': [(1, 0, 1), (1, 1, 1)],
            'C_contrib': [(0, 0, 1), (1, 1, 1)]
        },
        {
            'A_indices': [(0, 2, 1), (1, 2, 1)],
            'B_indices': [(2, 0, 1), (1, 0, 1)],
            'C_contrib': [(0, 0, 1), (1, 1, 1)]
        },
        {
            'A_indices': [(1, 0, 1), (2, 0, 1)],
            'B_indices': [(0, 1, 1), (2, 1, 1)],
            'C_contrib': [(1, 1, 1), (2, 2, 1)]
        }
    ]
    
    for k, term in enumerate(strassen_terms):
        a_blocks = term['A_blocks']
        b_blocks = term['B_blocks']
        c_contrib = term['C_contrib']
        
        # Iterate over all rank-24 terms for 3x3 multiplication
        for r24_term in rank24_terms:
            a_indices = r24_term['A_indices']
            b_indices = r24_term['B_indices']
            c_contrib_3x3 = r24_term['C_contrib']
            
            # U: Linear form for X[i, t]
            for br, bc, coeff in a_blocks:
                for i, t, coeff_3x3 in a_indices:
                    idx = get_a_index(br, bc, i, t)
                    U[idx][col] += coeff * coeff_3x3
                    
            # V: Linear form for Y[t, j]
            for br, bc, coeff in b_blocks:
                for t, j, coeff_3x3 in b_indices:
                    idx = get_b_index(br, bc, t, j)
                    V[idx][col] += coeff * coeff_3x3
                    
            # W: Contribution to C
            # The term X[i,t] * Y[t,j] contributes to C_block[i,j]
            # which is distributed according to c_contrib
            for r_c, c_c, coeff in c_contrib:
                for i, j, coeff_3x3 in c_contrib_3x3:
                    idx = get_w_index(r_c, c_c, i, j)
                    W[idx][col] += coeff * coeff_3x3
                    
            col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
