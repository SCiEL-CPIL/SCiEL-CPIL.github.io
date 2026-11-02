import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank estimate: 6 Strassen blocks (7 each) + 1 Outer Product (12) = 54
    rank = 54
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    current_col = 0
    
    def add_strassen_block(a_col_start, b_row_start, b_col_start):
        nonlocal current_col
        # a_row_start is always 0 for n=2
        a_row_start = 0
        a_row_end = 2
        b_row_end = b_row_start + 2
        b_col_end = b_col_start + 2
        
        # Local indices mapping
        # A[0, a_col_start], A[0, a_col_start+1]
        # A[1, a_col_start], A[1, a_col_start+1]
        # B[b_row_start, b_col_start], ...
        
        # Define the 7 Strassen terms
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Output mapping for the 2x2 block:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We iterate through the 7 terms and add their contribution to U, V, W
        
        # Term 1: M1
        # U: A[0, j0] + A[1, j1]
        # V: B[k0, l0] + B[k1, l1]
        # W: C[i0, l0] + 1, C[i1, l1] + 1
        
        terms = [
            {
                "name": "M1",
                "A": [(0, 0, 1.0), (1, 1, 1.0)], # (local_row, local_col, coeff)
                "B": [(0, 0, 1.0), (1, 1, 1.0)],
                "C": [(0, 0, 1.0), (1, 1, 1.0)]  # (local_row, local_col, coeff)
            },
            {
                "name": "M2",
                "A": [(1, 0, 1.0), (1, 1, 1.0)],
                "B": [(0, 0, 1.0)],
                "C": [(1, 0, 1.0), (1, 1, -1.0)]
            },
            {
                "name": "M3",
                "A": [(0, 0, 1.0)],
                "B": [(0, 1, 1.0), (1, 1, -1.0)],
                "C": [(0, 1, 1.0), (1, 1, 1.0)]
            },
            {
                "name": "M4",
                "A": [(1, 1, 1.0)],
                "B": [(1, 0, 1.0), (0, 0, -1.0)],
                "C": [(0, 0, 1.0), (1, 0, 1.0)]
            },
            {
                "name": "M5",
                "A": [(0, 0, 1.0), (0, 1, 1.0)],
                "B": [(1, 1, 1.0)],
                "C": [(0, 0, -1.0)]
            },
            {
                "name": "M6",
                "A": [(1, 0, 1.0), (0, 0, -1.0)],
                "B": [(0, 0, 1.0), (0, 1, 1.0)],
                "C": [(1, 1, 1.0)]
            },
            {
                "name": "M7",
                "A": [(0, 1, 1.0), (1, 1, -1.0)],
                "B": [(1, 0, 1.0), (1, 1, 1.0)],
                "C": [(0, 0, 1.0)]
            }
        ]
        
        for t in terms:
            col = current_col
            # Add U
            for lr, lc, coeff in t["A"]:
                global_row = (a_row_start + lr) * m + (a_col_start + lc)
                U[global_row][col] += coeff
            
            # Add V
            for lr, lc, coeff in t["B"]:
                global_row = (b_row_start + lr) * p + (b_col_start + lc)
                V[global_row][col] += coeff
            
            # Add W
            # W index: C[i, l] -> row l * n + i
            for lr, lc, coeff in t["C"]:
                global_i = a_row_start + lr
                global_l = b_col_start + lc
                global_row = global_l * n + global_i
                W[global_row][col] += coeff
                
            current_col += 1

    def add_outer_product_block(a_col, b_row):
        nonlocal current_col
        # A is 2x1 at column a_col
        # B is 1x6 at row b_row
        # C is 2x6
        
        for i in range(n):
            for l in range(p):
                col = current_col
                # U: A[i, a_col]
                global_u_row = i * m + a_col
                U[global_u_row][col] = 1.0
                
                # V: B[b_row, l]
                global_v_row = b_row * p + l
                V[global_v_row][col] = 1.0
                
                # W: C[i, l]
                global_w_row = l * n + i
                W[global_w_row][col] = 1.0
                
                current_col += 1

    # Define the blocks
    # A columns: 0,1 (Block A1), 2,3 (Block A2), 4 (Block A3)
    # B rows: 0,1 (Block B1), 2,3 (Block B2), 4 (Block B3)
    
    # Strassen Blocks:
    # 1. A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 0, 0)
    # 2. A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 0, 2)
    # 3. A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_block(0, 0, 4)
    
    # 4. A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen_block(2, 2, 0)
    # 5. A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(2, 2, 2)
    # 6. A[0:2, 2:4] * B[2:4, 4:6]
    add_strassen_block(2, 2, 4)
    
    # Outer Product Block:
    # 7. A[0:2, 4:5] * B[4:5, 0:6]
    add_outer_product_block(4, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
