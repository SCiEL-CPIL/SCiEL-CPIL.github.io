"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Rank 28 decomposition
    # Split inner dimension m=3 into 2 and 1.
    # A = [A1 | A2], B = [B1; B2]
    # A1 is 2x2, A2 is 2x1
    # B1 is 2x5, B2 is 1x5
    # C = A1*B1 + A2*B2
    
    # A1*B1 is 2x2 * 2x5.
    # Split B1 columns into 2, 2, 1.
    # B1_1 (cols 0,1): 2x2 * 2x2 -> Strassen rank 7
    # B1_2 (cols 2,3): 2x2 * 2x2 -> Strassen rank 7
    # B1_3 (col 4):    2x2 * 2x1 -> Schoolbook rank 4
    # Total for A1*B1: 18
    
    # A2*B2 is 2x1 * 1x5.
    # Schoolbook rank 2*1*5 = 10.
    # Total rank: 18 + 10 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for u_idx, u_val in zip(u_indices, u_vals):
            U[u_idx][col] = u_val
        for v_idx, v_val in zip(v_indices, v_vals):
            V[v_idx][col] = v_val
        for w_idx, w_val in zip(w_indices, w_vals):
            W[w_idx][col] = w_val
        col += 1
    
    # Indices for A1 (rows 0,1 of A, cols 0,1 of A)
    # U index = i*3 + j
    u_00 = 0
    u_01 = 1
    u_10 = 3
    u_11 = 4
    
    # Indices for B1 (rows 0,1 of B)
    # V index = j*5 + k
    
    # Indices for C (cols 0..4 of C, rows 0,1)
    # W index = k*2 + i
    
    # --- Block 1a: A1 * B1[:, 0:2] (Strassen) ---
    # B indices for cols 0,1
    v_00 = 0
    v_01 = 1
    v_10 = 5
    v_11 = 6
    
    # W indices for cols 0,1
    w_00 = 0
    w_01 = 1
    w_10 = 2
    w_11 = 3
    
    # M1 = (A00+A11)(B00+B11)
    add_term([u_00, u_11], [1, 1], 
             [v_00, v_11], [1, 1], 
             [w_00, w_11], [1, 1])
             
    # M2 = (A10+A11)B00
    add_term([u_10, u_11], [1, 1], 
             [v_00], [1], 
             [w_10], [1])
             
    # M3 = A00(B10-B11)
    add_term([u_00], [1], 
             [v_10, v_11], [1, -1], 
             [w_01, w_11], [1, 1])
             
    # M4 = A11(B01-B00)
    add_term([u_11], [1], 
             [v_01, v_00], [1, -1], 
             [w_00, w_10], [1, 1])
             
    # M5 = (A00+A01)B11
    add_term([u_00, u_01], [1, 1], 
             [v_11], [1], 
             [w_00, w_01], [-1, 1])
             
    # M6 = (A10-A00)(B00+B01)
    add_term([u_10, u_00], [1, -1], 
             [v_00, v_01], [1, 1], 
             [w_11], [1])
             
    # M7 = (A01-A11)(B10+B11)
    add_term([u_01, u_11], [1, -1], 
             [v_10, v_11], [1, 1], 
             [w_00], [1])

    # --- Block 1b: A1 * B1[:, 2:4] (Strassen) ---
    # B indices for cols 2,3
    v_02 = 2
    v_03 = 3
    v_12 = 7
    v_13 = 8
    
    # W indices for cols 2,3
    w_02 = 4
    w_03 = 5
    w_12 = 6
    w_13 = 7
    
    # M1
    add_term([u_00, u_11], [1, 1], 
             [v_02, v_13], [1, 1], 
             [w_02, w_13], [1, 1])
             
    # M2
    add_term([u_10, u_11], [1, 1], 
             [v_02], [1], 
             [w_12], [1])
             
    # M3
    add_term([u_00], [1], 
             [v_12, v_13], [1, -1], 
             [w_03, w_13], [1, 1])
             
    # M4
    add_term([u_11], [1], 
             [v_03, v_02], [1, -1], 
             [w_02, w_12], [1, 1])
             
    # M5
    add_term([u_00, u_01], [1, 1], 
             [v_13], [1], 
             [w_02, w_03], [-1, 1])
             
    # M6
    add_term([u_10, u_00], [1, -1], 
             [v_02, v_03], [1, 1], 
             [w_13], [1])
             
    # M7
    add_term([u_01, u_11], [1, -1], 
             [v_12, v_13], [1, 1], 
             [w_02], [1])

    # --- Block 1c: A1 * B1[:, 4] (Schoolbook) ---
    # B indices for col 4
    v_04 = 4
    v_14 = 9
    
    # W indices for col 4
    w_04 = 8
    w_14 = 9
    
    # A00 * B04 -> C04
    add_term([u_00], [1], 
             [v_04], [1], 
             [w_04], [1])
             
    # A01 * B14 -> C04
    add_term([u_01], [1], 
             [v_14], [1], 
             [w_04], [1])
             
    # A10 * B04 -> C14
    add_term([u_10], [1], 
             [v_04], [1], 
             [w_14], [1])
             
    # A11 * B14 -> C14
    add_term([u_11], [1], 
             [v_14], [1], 
             [w_14], [1])

    # --- Block 2: A2 * B2 (Schoolbook) ---
    # A2 indices: col 2 of A
    u_02 = 2
    u_12 = 5
    
    # B2 indices: row 2 of B
    # V index = 2*5 + k = 10 + k
    
    # W indices: all k
    
    for k in range(5):
        v_2k = 10 + k
        w_0k = 2 * k
        w_1k = 2 * k + 1
        
        # A02 * B2k -> C0k
        add_term([u_02], [1], 
                 [v_2k], [1], 
                 [w_0k], [1])
                 
        # A12 * B2k -> C1k
        add_term([u_12], [1], 
                 [v_2k], [1], 
                 [w_1k], [1])

    # Check if we used all columns
    # assert col == rank, f"Column mismatch: {col} vs {rank}"
    
    if time.time() > deadline - 5:
        # If we are close to deadline, just return what we have (should be complete)
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
