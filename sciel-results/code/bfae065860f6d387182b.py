import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of columns
    # Or just accumulate into arrays
    
    # We use the decomposition:
    # Split p=8 into p1=2, p2=3, p3=3
    # Block 1: M_{2,3,2} using M_{2,2,2} (Strassen) + M_{2,1,2} (Schoolbook)
    # Block 2: M_{2,3,3} using M_{2,2,3} (M_{2,2,2} Strassen + M_{2,2,1} Schoolbook) + M_{2,1,3} (Schoolbook)
    # Block 3: M_{2,3,3} same as Block 2
    
    # Total Rank = 11 + 17 + 17 = 45
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict {(i,j): val}
        # v_coeffs: dict {(j,k): val}
        # w_coeffs: dict {(i,k): val}
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for (i, j), val in u_coeffs.items():
            u_vec[i * m + j] += val
        for (j, k), val in v_coeffs.items():
            v_vec[j * p + k] += val
        for (i, k), val in w_coeffs.items():
            w_vec[k * n + i] += val
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # --- Block 1: p_offset = 0, p_size = 2 ---
    # A is 2x3, B is 3x2.
    # Split A into A_22 (cols 0,1) and A_1 (col 2).
    # Split B into B_22 (rows 0,1) and B_1 (row 2).
    # C = A_22 B_22 + A_1 B_1.
    
    # Sub-block 1.1: A_22 (2x2) * B_22 (2x2) -> Strassen
    # Indices: i in 0..1, j in 0..1, k in 0..1 (offset 0)
    
    # M1
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    # M2
    add_term({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1})
    # M3
    add_term({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1})
    # M4
    add_term({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,1):1})
    # M5
    add_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1})
    # M6
    add_term({(1,0):-1, (0,0):1}, {(0,0):1, (0,1):1}, {(1,1):1})
    # M7
    add_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1})
    
    # Sub-block 1.2: A_1 (2x1, col 2) * B_1 (1x2, row 2) -> Schoolbook
    # A index (i, 2). B index (2, k). C index (i, k).
    for i in range(2):
        for k in range(2):
            add_term({(i, 2): 1}, {(2, k): 1}, {(i, k): 1})

    # --- Block 2: p_offset = 2, p_size = 3 ---
    # A is 2x3, B is 3x3 (cols 2..4 of original B? No, B is 3x8. Block 2 is cols 2,3,4).
    # Wait, B is 3x8.
    # Block 1 used B cols 0,1.
    # Block 2 uses B cols 2,3,4.
    # Block 3 uses B cols 5,6,7.
    
    # A is shared (2x3).
    # Split A into A_22 (cols 0,1) and A_1 (col 2).
    # Split B_block (3x3) into B_23 (rows 0,1, cols 0,1,2 of block) and B_1 (row 2).
    # Wait, B_block is 3x3.
    # Rows of B_block correspond to j=0,1,2.
    # Cols of B_block correspond to k=2,3,4.
    
    # C_block = A_22 B_23 + A_1 B_1.
    
    # Sub-block 2.1: A_22 (2x2) * B_23 (2x3).
    # Split B_23 into B_22 (cols 0,1 of block, i.e., k=2,3) and B_21 (col 2 of block, i.e., k=4).
    # A_22 * B_22 -> Strassen.
    # A_22 * B_21 -> Schoolbook.
    
    # Strassen part (k=2,3)
    # Map local k 0->2, 1->3.
    # M1
    add_term({(0,0):1, (1,1):1}, {(0,2):1, (1,3):1}, {(0,2):1, (1,3):1})
    # M2
    add_term({(1,0):1, (1,1):1}, {(0,2):1}, {(1,2):1})
    # M3
    add_term({(0,0):1}, {(0,3):1, (1,3):-1}, {(0,3):1})
    # M4
    add_term({(1,1):1}, {(1,2):1, (0,2):-1}, {(0,2):1, (1,3):1})
    # M5
    add_term({(0,0):1, (0,1):1}, {(1,3):1}, {(0,2):-1, (0,3):1})
    # M6
    add_term({(1,0):-1, (0,0):1}, {(0,2):1, (0,3):1}, {(1,3):1})
    # M7
    add_term({(0,1):1, (1,1):-1}, {(1,2):1, (1,3):1}, {(0,2):1, (1,3):-1})
    
    # Schoolbook part (k=4)
    # A_22 (i in 0,1; j in 0,1) * B_21 (j in 0,1; k=4)
    for i in range(2):
        for j in range(2):
            for k in [4]:
                add_term({(i, j): 1}, {(j, k): 1}, {(i, k): 1})
                
    # Sub-block 2.2: A_1 (2x1, col 2) * B_1 (1x3, row 2, cols 2,3,4).
    # A index (i, 2). B index (2, k). C index (i, k).
    for i in range(2):
        for k in [2, 3, 4]:
            add_term({(i, 2): 1}, {(2, k): 1}, {(i, k): 1})

    # --- Block 3: p_offset = 5, p_size = 3 ---
    # Same as Block 2, but k = 5, 6, 7.
    
    # Strassen part (k=5,6)
    # M1
    add_term({(0,0):1, (1,1):1}, {(0,5):1, (1,6):1}, {(0,5):1, (1,6):1})
    # M2
    add_term({(1,0):1, (1,1):1}, {(0,5):1}, {(1,5):1})
    # M3
    add_term({(0,0):1}, {(0,6):1, (1,6):-1}, {(0,6):1})
    # M4
    add_term({(1,1):1}, {(1,5):1, (0,5):-1}, {(0,5):1, (1,6):1})
    # M5
    add_term({(0,0):1, (0,1):1}, {(1,6):1}, {(0,5):-1, (0,6):1})
    # M6
    add_term({(1,0):-1, (0,0):1}, {(0,5):1, (0,6):1}, {(1,6):1})
    # M7
    add_term({(0,1):1, (1,1):-1}, {(1,5):1, (1,6):1}, {(0,5):1, (1,6):-1})
    
    # Schoolbook part (k=7)
    for i in range(2):
        for j in range(2):
            for k in [7]:
                add_term({(i, j): 1}, {(j, k): 1}, {(i, k): 1})
                
    # Sub-block 3.2: A_1 * B_1 (k=5,6,7)
    for i in range(2):
        for k in [5, 6, 7]:
            add_term({(i, 2): 1}, {(2, k): 1}, {(i, k): 1})

    rank = len(U_cols)
    # Convert to required format
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
