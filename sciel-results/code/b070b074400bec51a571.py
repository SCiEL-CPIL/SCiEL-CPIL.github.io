"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_2x2x2(a_rows, b_rows, w_rows):
        nonlocal col
        # a_rows: list of 2 indices for A
        # b_rows: list of 2 indices for B
        # w_rows: list of 4 indices for W (k*n+i)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[a_rows[i]][col] = 1
                    V[b_rows[j]][col] = 1
                    W[w_rows[k*2+i]][col] = 1
                    col += 1

    def add_1x2x2(a_row, b_rows, w_rows):
        nonlocal col
        # a_row: index for A
        # b_rows: list of 2 indices for B
        # w_rows: list of 2 indices for W (k*n+i)
        for j in range(2):
            for k in range(2):
                U[a_row][col] = 1
                V[b_rows[j]][col] = 1
                W[w_rows[k]][col] = 1
                col += 1

    # Top part: A rows 0,1. B rows 0-3. C cols 0-5.
    # W rows for C cols 0-1: k=0,1; i=0,1 -> 0,1,3,4
    # W rows for C cols 2-3: k=0,1; i=0,1 -> 0,1,3,4? No.
    # C is 3x6. C[i,k].
    # W index is k*n+i.
    # For C cols 0-1 (k=0,1), rows i=0,1 (top part).
    # Indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4.
    # For C cols 2-3 (k=2,3), rows i=0,1.
    # Indices: 2*3+0=6, 2*3+1=7, 3*3+0=9, 3*3+1=10.
    # For C cols 4-5 (k=4,5), rows i=0,1.
    # Indices: 4*3+0=12, 4*3+1=13, 5*3+0=15, 5*3+1=16.
    
    # A rows 0,1. A cols 0-1 (rows 0,1 of A), A cols 2-3 (rows 2,3 of A).
    # B rows 0-1 (rows 0,1 of B), B rows 2-3 (rows 2,3 of B).
    
    # Block 1: C cols 0-1. B rows 0-1.
    # A cols 0-1: A rows 0,1. B rows 0,1. W rows 0,1,3,4.
    add_2x2x2([0, 1], [0, 1], [0, 1, 3, 4])
    # A cols 2-3: A rows 2,3. B rows 0,1. W rows 0,1,3,4.
    add_2x2x2([2, 3], [0, 1], [0, 1, 3, 4])
    
    # Block 2: C cols 2-3. B rows 2-3.
    # A cols 0-1: A rows 0,1. B rows 2,3. W rows 6,7,9,10.
    add_2x2x2([0, 1], [2, 3], [6, 7, 9, 10])
    # A cols 2-3: A rows 2,3. B rows 2,3. W rows 6,7,9,10.
    add_2x2x2([2, 3], [2, 3], [6, 7, 9, 10])
    
    # Block 3: C cols 4-5. B rows 4-5.
    # A cols 0-1: A rows 0,1. B rows 4,5. W rows 12,13,15,16.
    add_2x2x2([0, 1], [4, 5], [12, 13, 15, 16])
    # A cols 2-3: A rows 2,3. B rows 4,5. W rows 12,13,15,16.
    add_2x2x2([2, 3], [4, 5], [12, 13, 15, 16])
    
    # Bottom part: A row 2. B rows 0-3. C cols 0-5.
    # W rows for C cols 0-1: k=0,1; i=2 -> 0*3+2=2, 1*3+2=5.
    # W rows for C cols 2-3: k=2,3; i=2 -> 2*3+2=8, 3*3+2=11.
    # W rows for C cols 4-5: k=4,5; i=2 -> 4*3+2=14, 5*3+2=17.
    
    # A row 2. A cols 0-1 (row 8 of A), A cols 2-3 (row 10 of A).
    # B rows 0-1, 2-3, 4-5.
    
    # Block 1: C cols 0-1. B rows 0-1.
    # A cols 0-1: A row 8. B rows 0,1. W rows 2,5.
    add_1x2x2(8, [0, 1], [2, 5])
    # A cols 2-3: A row 10. B rows 0,1. W rows 2,5.
    add_1x2x2(10, [0, 1], [2, 5])
    
    # Block 2: C cols 2-3. B rows 2-3.
    # A cols 0-1: A row 8. B rows 2,3. W rows 8,11.
    add_1x2x2(8, [2, 3], [8, 11])
    # A cols 2-3: A row 10. B rows 2,3. W rows 8,11.
    add_1x2x2(10, [2, 3], [8, 11])
    
    # Block 3: C cols 4-5. B rows 4-5.
    # A cols 0-1: A row 8. B rows 4,5. W rows 14,17.
    add_1x2x2(8, [4, 5], [14, 17])
    # A cols 2-3: A row 10. B rows 4,5. W rows 14,17.
    add_1x2x2(10, [4, 5], [14, 17])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
