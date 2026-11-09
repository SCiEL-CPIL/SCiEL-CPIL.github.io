"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm on 2x2 blocks of 3x3 matrices.
    # Rank = 7 (Strassen) * 27 (Schoolbook 3x3) = 189.
    rank = 189
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definition
    # Each term is a tuple: (x_blocks, y_blocks, c_blocks)
    # x_blocks: list of (block_id, coeff)
    # y_blocks: list of (block_id, coeff)
    # c_blocks: list of (block_id, coeff)
    # Block IDs: 11, 12, 21, 22
    
    strassen_terms = [
        # M1 = (A11 + A22)(B11 + B22)
        ([ (11, 1), (22, 1) ], [ (11, 1), (22, 1) ], [ (11, 1), (22, 1) ]),
        # M2 = (A21 + A22)B11
        ([ (21, 1), (22, 1) ], [ (11, 1) ], [ (21, 1), (22, -1) ]),
        # M3 = A11(B12 - B22)
        ([ (11, 1) ], [ (12, 1), (22, -1) ], [ (12, 1), (22, 1) ]),
        # M4 = A22(B21 - B11)
        ([ (22, 1) ], [ (21, 1), (11, -1) ], [ (11, 1), (21, 1) ]),
        # M5 = (A11 + A12)B22
        ([ (11, 1), (12, 1) ], [ (22, 1) ], [ (11, -1), (12, 1) ]),
        # M6 = (A21 - A11)(B11 + B12)
        ([ (21, 1), (11, -1) ], [ (11, 1), (12, 1) ], [ (22, 1) ]),
        # M7 = (A12 - A22)(B21 + B22)
        ([ (12, 1), (22, -1) ], [ (21, 1), (22, 1) ], [ (11, 1) ]),
    ]
    
    # Helper to get global index from block id and local indices
    # Block 11: offset (0,0), Block 12: offset (0,3), Block 21: offset (3,0), Block 22: offset (3,3)
    def get_offset(block_id):
        if block_id == 11: return (0, 0)
        if block_id == 12: return (0, 3)
        if block_id == 21: return (3, 0)
        if block_id == 22: return (3, 3)
        return (0, 0)
    
    col = 0
    for x_blocks, y_blocks, c_blocks in strassen_terms:
        # For each Strassen term, we perform a 3x3 schoolbook multiplication
        # Schoolbook indices i, j, k in 0..2
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Update U (Input A)
                    # U index: row * 6 + col
                    # Local A index for term (i,j) in block is (i, j)
                    for b_id, coeff in x_blocks:
                        r_off, c_off = get_offset(b_id)
                        u_idx = (r_off + i) * 6 + (c_off + j)
                        U[u_idx][col] += coeff
                        
                    # Update V (Input B)
                    # Local B index for term (j,k) in block is (j, k)
                    for b_id, coeff in y_blocks:
                        r_off, c_off = get_offset(b_id)
                        v_idx = (r_off + j) * 6 + (c_off + k)
                        V[v_idx][col] += coeff
                        
                    # Update W (Output C)
                    # Local C index for term (k,i) in block is (k, i)
                    # Convention: W[k*n+i] -> Row k, Col i
                    for b_id, coeff in c_blocks:
                        r_off, c_off = get_offset(b_id)
                        w_idx = (r_off + k) * 6 + (c_off + i)
                        W[w_idx][col] += coeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
