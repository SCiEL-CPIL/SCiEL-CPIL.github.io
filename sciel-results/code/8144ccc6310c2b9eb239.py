import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 92 decomposition
    rank = 92
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(i_u, j_u, i_v, j_v, i_w, j_w, val=1):
        nonlocal col
        U[i_u * m + j_u][col] = val
        V[i_v * p + j_v][col] = val
        W[i_w * n + j_w][col] = val
        col += 1
        
    # We need to map local indices to global.
    # Global A is 4x5. Global B is 5x5. Global C is 5x4.
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Type: 2x2x2 Strassen
    # Local indices 0,1 map to Global 0,1
    add_strassen_block(0, 0, 0, 2, 2, 2)
    
    # Block 2: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # Type: 2x3x2
    # A cols 2,3,4. B rows 2,3,4.
    # Split B rows into 1 (row 2) and 2 (rows 3,4)? Or A cols?
    # Let's split the dimension 3 (m or p) into 1+2.
    # Here m=3 (cols of A), p=2 (cols of C).
    # Split m into 1+2.
    # Part 1: A[0:2, 2:3] * B[2:3, 0:2] -> 2x1x2 Schoolbook
    # Part 2: A[0:2, 3:5] * B[3:5, 0:2] -> 2x2x2 Strassen
    add_schoolbook_block(0, 2, 2, 2, 1, 2)
    add_strassen_block(0, 3, 3, 2, 2, 2)
    
    # Block 3: A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    # Type: 2x2x2 Strassen
    add_strassen_block(2, 0, 0, 2, 2, 2)
    
    # Block 4: A[2:4, 2:5] * B[2:5, 0:2] -> C[2:4, 0:2]
    # Type: 2x3x2
    # Split m into 1+2.
    add_schoolbook_block(2, 2, 2, 2, 1, 2)
    add_strassen_block(2, 3, 3, 2, 2, 2)
    
    # Block 5: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # Type: 2x2x3
    # Split p into 1+2.
    # Part 1: A[0:2, 0:2] * B[0:2, 2:3] -> 2x2x1 Schoolbook
    # Part 2: A[0:2, 0:2] * B[0:2, 3:5] -> 2x2x2 Strassen
    add_schoolbook_block(0, 0, 0, 2, 2, 1)
    add_strassen_block(0, 0, 3, 2, 2, 2)
    
    # Block 6: A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    # Type: 2x3x3
    # Split m into 1+2.
    # Part 1: A[0:2, 2:3] * B[2:3, 2:5] -> 2x1x3 Schoolbook
    # Part 2: A[0:2, 3:5] * B[3:5, 2:5] -> 2x2x3 (Split p into 1+2)
    #   Part 2a: 2x2x1 Schoolbook
    #   Part 2b: 2x2x2 Strassen
    add_schoolbook_block(0, 2, 2, 2, 1, 3)
    add_schoolbook_block(0, 3, 3, 2, 2, 1)
    add_strassen_block(0, 3, 4, 2, 2, 2)
    
    # Block 7: A[2:4, 0:2] * B[0:2, 2:5] -> C[2:4, 2:5]
    # Type: 2x2x3
    # Split p into 1+2.
    add_schoolbook_block(2, 0, 0, 2, 2, 1)
    add_strassen_block(2, 0, 3, 2, 2, 2)
    
    # Block 8: A[2:4, 2:5] * B[2:5, 2:5] -> C[2:4, 2:5]
    # Type: 2x3x3
    # Split m into 1+2.
    add_schoolbook_block(2, 2, 2, 2, 1, 3)
    add_schoolbook_block(2, 3, 3, 2, 2, 1)
    add_strassen_block(2, 3, 4, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}

# Implement helper functions
