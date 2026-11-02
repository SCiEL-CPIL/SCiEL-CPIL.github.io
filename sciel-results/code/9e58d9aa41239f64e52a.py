import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We know rank 28 is achievable via 4 Strassen blocks.
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a Strassen decomposition for a 2x2 block
    def add_strassen_block(a_offset, b_row_offset, b_col_offset, c_offset, start_col):
        # A is 2x2, indices a_offset + [0,1,2,3]
        # B is 2x2, indices (b_row_offset + r)*p + (b_col_offset + c)
        # C is 2x2, indices c_offset + [0,1,2,3]
        
        # Strassen products M1..M7
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Coefficients for A (U), B (V), C (W)
        # Indices in local 2x2:
        # A: 00=0, 01=1, 10=2, 11=3
        # B: 00=0, 01=1, 10=2, 11=3
        # C: 00=0, 01=1, 10=2, 11=3
        
        products = [
            # (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
            ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]), # M1
            ([2, 3], [1, 1], [0], [1], [2], [1]),              # M2 -> C10
            ([0], [1], [1, 3], [1, -1], [1], [1]),             # M3 -> C01
            ([3], [1], [2, 0], [1, -1], [2], [1]),             # M4 -> C10 (Wait, M4 is C00 and C11? No)
            # Let's re-verify Strassen formulas
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # M1: C00 (+1), C11 (+1)
            # M2: C10 (+1)
            # M3: C01 (+1)
            # M4: C00 (+1), C10 (+1)
            # M5: C00 (-1), C01 (+1)
            # M6: C11 (+1)
            # M7: C00 (+1)
        ]
        
        # Corrected W contributions:
        # M1: W[0]+=1, W[3]+=1
        # M2: W[2]+=1
        # M3: W[1]+=1
        # M4: W[0]+=1, W[2]+=1
        # M5: W[0]-=1, W[1]+=1
        # M6: W[3]+=1
        # M7: W[0]+=1
        
        w_contribs = [
            {0: 1, 3: 1}, # M1
            {2: 1},       # M2
            {1: 1},       # M3
            {0: 1, 2: 1}, # M4
            {0: -1, 1: 1},# M5
            {3: 1},       # M6
            {0: 1}        # M7
        ]
        
        u_indices_vals = [
            ([0, 3], [1, 1]), # M1
            ([2, 3], [1, 1]), # M2
            ([0], [1]),       # M3
            ([3], [1]),       # M4
            ([0, 1], [1, 1]), # M5
            ([2, 0], [1, -1]),# M6
            ([1, 3], [1, -1]) # M7
        ]
        
        v_indices_vals = [
            ([0, 3], [1, 1]), # M1
            ([0], [1]),       # M2
            ([1, 3], [1, -1]),# M3
            ([2, 0], [1, -1]),# M4
            ([3], [1]),       # M5
            ([0, 1], [1, 1]), # M6
            ([2, 3], [1, 1])  # M7
        ]
        
        for k in range(7):
            col = start_col + k
            
            # Fill U
            u_idx, u_val = u_indices_vals[k]
            for idx, val in zip(u_idx, u_val):
                U[a_offset + idx, col] += val
                
            # Fill V
            v_idx, v_val = v_indices_vals[k]
            for idx, val in zip(v_idx, v_val):
                # B index: row = b_row_offset + r, col = b_col_offset + c
                # Local idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                r = idx // 2
                c = idx % 2
                global_b_idx = (b_row_offset + r) * p + (b_col_offset + c)
                V[global_b_idx, col] += val
                
            # Fill W
            w_map = w_contribs[k]
            for idx, val in w_map.items():
                W[c_offset + idx, col] += val
                
    # Block 1: A(0,1) * B(0,1; 0,1) -> C(0,1)
    # A offset 0, B row 0, B col 0, C offset 0
    add_strassen_block(0, 0, 0, 0, 0)
    
    # Block 2: A(2,3) * B(2,3; 0,1) -> C(0,1)
    # A offset 2, B row 2, B col 0, C offset 0
    add_strassen_block(2, 2, 0, 0, 7)
    
    # Block 3: A(0,1) * B(0,1; 2,3) -> C(2,3)
    # A offset 0, B row 0, B col 2, C offset 2
    add_strassen_block(0, 0, 2, 2, 14)
    
    # Block 4: A(2,3) * B(2,3; 2,3) -> C(2,3)
    # A offset 2, B row 2, B col 2, C offset 2
    add_strassen_block(2, 2, 2, 2, 21)
    
    # Convert to list of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
