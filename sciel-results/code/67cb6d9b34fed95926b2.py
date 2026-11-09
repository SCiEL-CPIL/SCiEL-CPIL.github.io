import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct the decomposition for M_{2,3,6}
    # Strategy: Split A into A_left (2x2) and A_right (2x1)
    # Split B into B_top (2x6) and B_bottom (1x6)
    # B_top is split into 3 blocks of 2x2: B_0, B_1, B_2
    
    # Part 1: A_left * B_top
    # This is 3 independent 2x2 multiplications.
    # We use Strassen for each.
    # Part 2: A_right * B_bottom
    # This is an outer product 2x1 * 1x6.
    
    # Total Rank = 3 * 7 (Strassen) + 12 (Outer) = 33
    
    rank = 33
    
    # Initialize U, V, W as lists of lists (to ensure exact integer/half-integer if needed, though Strassen uses ints)
    # U: (n*m, rank) = (6, 33)
    # V: (m*p, rank) = (18, 33)
    # W: (p*n, rank) = (12, 33)
    
    U = np.zeros((6, rank), dtype=float)
    V = np.zeros((18, rank), dtype=float)
    W = np.zeros((12, rank), dtype=float)
    
    term_idx = 0
    
    # --- Part 1: Strassen blocks ---
    # Indices in U for A_left (2x2 part of 2x3 A)
    # A is 2x3. A_left is cols 0,1.
    # U indices:
    # 0: A[0,0]
    # 1: A[0,1]
    # 2: A[0,2] (used in Part 2)
    # 3: A[1,0]
    # 4: A[1,1]
    # 5: A[1,2] (used in Part 2)
    
    a_idx = {
        'A00': 0, 'A01': 1, 
        'A10': 3, 'A11': 4
    }
    
    # Strassen coefficients for 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_defs = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: dict {u_idx: coeff}
        # v_coeffs: dict {v_local_idx: coeff}
        # w_coeffs: dict {w_local_idx: coeff}
        
        # M1
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1}),
        # M2
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1, 'C11': -1}),
        # M3
        ({'A00': 1}, {'B01': 1, 'B11': -1}, {'C01': 1, 'C11': 1}),
        # M4
        ({'A11': 1}, {'B10': 1, 'B00': -1}, {'C00': 1, 'C10': 1}),
        # M5
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': -1, 'C01': 1}),
        # M6
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B01': 1}, {'C11': 1}),
        # M7
        ({'A01': 1, 'A11': -1}, {'B10': 1, 'B11': 1}, {'C00': 1})
    ]
    
    for j in range(3):
        # j is the block index (0, 1, 2)
        # B_block is B_top[0:2, 2*j:2*j+2]
        
        # V indices for this block
        # B_top rows 0,1.
        # B[0, 2j] -> V idx 0*6 + 2j = 2j
        # B[0, 2j+1] -> V idx 0*6 + 2j+1 = 2j+1
        # B[1, 2j] -> V idx 1*6 + 2j = 6+2j
        # B[1, 2j+1] -> V idx 1*6 + 2j+1 = 6+2j+1
        
        v_map = {
            'B00': 2*j,
            'B01': 2*j+1,
            'B10': 6+2*j,
            'B11': 6+2*j+1
        }
        
        # W indices for this block
        # C block is C[:, 2*j : 2*j+2]
        # C[0, 2j] -> W idx (2j)*2 + 0 = 4j
        # C[0, 2j+1] -> W idx (2j+1)*2 + 0 = 4j+2
        # C[1, 2j] -> W idx (2j)*2 + 1 = 4j+1
        # C[1, 2j+1] -> W idx (2j+1)*2 + 1 = 4j+3
        
        w_map = {
            'C00': 4*j,
            'C01': 4*j+2,
            'C10': 4*j+1,
            'C11': 4*j+3
        }
        
        for u_def, v_def, w_def in strassen_defs:
            # Fill U
            for key, val in u_def.items():
                U[a_idx[key], term_idx] += val
            
            # Fill V
            for key, val in v_def.items():
                V[v_map[key], term_idx] += val
            
            # Fill W
            for key, val in w_def.items():
                W[w_map[key], term_idx] += val
                
            term_idx += 1
            
    # --- Part 2: Outer Product A_right * B_bottom ---
    # A_right is cols 2 of A (indices 2, 5 in U)
    # B_bottom is row 2 of B (indices 12..17 in V)
    # Output is all C (indices 0..11 in W)
    
    # Terms: A[i,2] * B[2,k] -> C[i,k]
    # i in 0..1, k in 0..5
    
    for i in range(2):
        for k in range(6):
            u_idx = i*3 + 2 # A[i,2]
            v_idx = 2*6 + k # B[2,k]
            w_idx = k*2 + i # C[k,i]
            
            U[u_idx, term_idx] = 1
            V[v_idx, term_idx] = 1
            W[w_idx, term_idx] = 1
            
            term_idx += 1
            
    # Convert to lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
