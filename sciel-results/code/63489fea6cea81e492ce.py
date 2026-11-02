import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # A indices in 2x2 block: 00, 01, 10, 11
    # B indices in 2x2 block: 00, 01, 10, 11
    
    # Coefficients for A in each M
    # M1: 1, 0, 0, 1
    # M2: 0, 0, 1, 1
    # M3: 1, 0, 0, 0
    # M4: 0, 0, 0, 1
    # M5: 1, 1, 0, 0
    # M6: -1, 0, 1, 0
    # M7: 0, 1, 0, -1
    A_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # Coefficients for B in each M
    # M1: 1, 0, 0, 1
    # M2: 1, 0, 0, 0
    # M3: 0, -1, 1, 0
    # M4: -1, 1, 0, 0
    # M5: 0, 0, 0, 1
    # M6: 1, 0, 1, 0
    # M7: 0, 1, 0, 1
    B_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, -1, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    
    # Coefficients for C in terms of M
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Define the 4 blocks
    # Block 0: A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    # Block 1: A[0:2, 2:4], B[2:4, 0:2], C[0:2, 2:4]
    # Block 2: A[0:2, 0:2], B[0:2, 2:4], C[0:2, 2:4] -- Wait, C1 is A1B11+A2B21.
    # C1 corresponds to cols 0,1 of C.
    # C2 corresponds to cols 2,3 of C.
    
    # Let's map blocks to (A_cols, B_rows, B_cols, C_cols)
    # C_col 0,1:
    #   Term 1: A_cols 0,1 (A1), B_rows 0,1 (B11), B_cols 0,1
    #   Term 2: A_cols 2,3 (A2), B_rows 2,3 (B21), B_cols 0,1
    # C_col 2,3:
    #   Term 3: A_cols 0,1 (A1), B_rows 0,1 (B12), B_cols 2,3
    #   Term 4: A_cols 2,3 (A2), B_rows 2,3 (B22), B_cols 2,3
    
    blocks = [
        {'a_cols': [0, 1], 'b_rows': [0, 1], 'b_cols': [0, 1], 'c_cols': [0, 1]},
        {'a_cols': [2, 3], 'b_rows': [2, 3], 'b_cols': [0, 1], 'c_cols': [0, 1]},
        {'a_cols': [0, 1], 'b_rows': [0, 1], 'b_cols': [2, 3], 'c_cols': [2, 3]},
        {'a_cols': [2, 3], 'b_rows': [2, 3], 'b_cols': [2, 3], 'c_cols': [2, 3]}
    ]
    
    for b_idx, block in enumerate(blocks):
        base_c = b_idx * 7
        a_cols = block['a_cols']
        b_rows = block['b_rows']
        b_cols = block['b_cols']
        c_cols = block['c_cols']
        
        # A indices in global A (2x4)
        # A[i, j]
        # We need to map local 2x2 indices (0,1) to global j.
        # a_cols are the global column indices.
        
        for s in range(7):
            c = base_c + s
            
            # Fill U
            for local_i in range(2):
                for local_j in range(2):
                    global_j = a_cols[local_j]
                    global_i = local_i
                    val = A_coeffs[s][local_j * 2 + local_i] # Wait, A_coeffs is indexed by M, then A_elem.
                    # My A_coeffs list is [M][A_elem]. A_elem is 00, 01, 10, 11.
                    # Index in A_coeffs[s] is local_j * 2 + local_i?
                    # 00: j=0, i=0 -> 0
                    # 01: j=1, i=0 -> 1
                    # 10: j=0, i=1 -> 2
                    # 11: j=1, i=1 -> 3
                    # So index = local_j * 2 + local_i.
                    U[global_i * m + global_j][c] = A_coeffs[s][local_j * 2 + local_i]
                    
            # Fill V
            for local_j in range(2):
                for local_k in range(2):
                    global_j = b_rows[local_j]
                    global_k = b_cols[local_k]
                    val = B_coeffs[s][local_j * 2 + local_k]
                    V[global_j * p + global_k][c] = B_coeffs[s][local_j * 2 + local_k]
                    
            # Fill W
            for local_i in range(2):
                for local_k in range(2):
                    global_i = local_i
                    global_k = c_cols[local_k]
                    # C_coeffs is [C_elem][M].
                    # C_elem index = local_k * 2 + local_i.
                    # We need to sum over M (s).
                    # So for each s, we add C_coeffs[C_elem][s] to W.
                    pass
            
            # Correct W filling logic:
            # W[global_i * n + global_k][c] should be the coefficient of C_global_i,global_k in term c.
            # Term c corresponds to Strassen term s of block b_idx.
            # The contribution to C_block is sum_s C_coeffs[C_elem][s] * M_s.
            # So W[global_i, global_k, c] = C_coeffs[C_elem][s].
            
            for local_i in range(2):
                for local_k in range(2):
                    global_i = local_i
                    global_k = c_cols[local_k]
                    c_elem_idx = local_k * 2 + local_i
                    W[global_k * n + global_i][c] = C_coeffs[c_elem_idx][s]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
