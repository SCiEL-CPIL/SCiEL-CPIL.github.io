import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Rank calculation: 6 instances of 2x2x2 Strassen (rank 7 each)
    # 6 * 7 = 42
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 decomposition terms
    # Local indices: A (u, v), B (v, w), C (w, u)
    # u, v, w in {0, 1}
    # Term structure: (coeff_A, u_A, v_A, coeff_B, v_B, w_B, coeff_C, w_C, u_C)
    # Actually, it's simpler to define the 7 multiplications M1..M7
    # and their contributions to C.
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We define the 7 "basis" tensors for 2x2x2 Strassen.
    # Each term t in 0..6 has specific non-zero entries in U, V, W.
    # Let's list the factors for each term.
    # Term 0 (M1): A part: (1,0,0), (1,1,1) ; B part: (1,0,0), (1,1,1) ; C part: (1,0,0), (0,1,1), (-1,1,1), (1,0,1)
    # Wait, C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the 7 Strassen components explicitly as lists of (u_idx, v_idx, coeff) for A, (v_idx, w_idx, coeff) for B, (w_idx, u_idx, coeff) for C.
    # Note: A is 2x2, indices 0,1 for row and col.
    # B is 2x2, indices 0,1 for row and col.
    # C is 2x2, indices 0,1 for row (w) and col (u).
    
    strassen_terms = []
    
    # M1
    strassen_terms.append({
        'U': [(0,0,1), (1,1,1)], # A11, A22
        'V': [(0,0,1), (1,1,1)], # B11, B22
        'W': [(0,0,1), (1,1,1), (1,1,-1), (0,1,1)] # C11(+1), C22(+1), C21(-1)? No.
        # Let's re-verify C indices.
        # C[w, u].
        # C11 is w=0, u=0. Coeff 1.
        # C22 is w=1, u=1. Coeff 1.
        # C21 is w=1, u=0. Coeff -1.
        # C12 is w=0, u=1. Coeff 1.
        # So W for M1: (0,0,1), (1,1,1), (1,0,-1), (0,1,1)
    })
    strassen_terms[-1]['W'] = [(0,0,1), (1,1,1), (1,0,-1), (0,1,1)]
    
    # M2: (A21+A22)B11
    # A: A21 (1,0), A22 (1,1)
    # B: B11 (0,0)
    # C: C21 (1,0) coeff 1, C22 (1,1) coeff -1
    strassen_terms.append({
        'U': [(1,0,1), (1,1,1)],
        'V': [(0,0,1)],
        'W': [(1,0,1), (1,1,-1)]
    })
    
    # M3: A11(B21-B22)
    # A: A11 (0,0)
    # B: B21 (1,0) coeff 1, B22 (1,1) coeff -1
    # C: C12 (0,1) coeff 1, C22 (1,1) coeff 1
    strassen_terms.append({
        'U': [(0,0,1)],
        'V': [(1,0,1), (1,1,-1)],
        'W': [(0,1,1), (1,1,1)]
    })
    
    # M4: A22(B12-B11)
    # A: A22 (1,1)
    # B: B12 (0,1) coeff 1, B11 (0,0) coeff -1
    # C: C11 (0,0) coeff 1, C21 (1,0) coeff 1
    strassen_terms.append({
        'U': [(1,1,1)],
        'V': [(0,1,1), (0,0,-1)],
        'W': [(0,0,1), (1,0,1)]
    })
    
    # M5: (A11+A12)B22
    # A: A11 (0,0), A12 (0,1)
    # B: B22 (1,1)
    # C: C11 (0,0) coeff -1, C12 (0,1) coeff 1
    strassen_terms.append({
        'U': [(0,0,1), (0,1,1)],
        'V': [(1,1,1)],
        'W': [(0,0,-1), (0,1,1)]
    })
    
    # M6: (A21-A11)(B11+B12)
    # A: A21 (1,0) coeff 1, A11 (0,0) coeff -1
    # B: B11 (0,0), B12 (0,1)
    # C: C22 (1,1) coeff 1
    strassen_terms.append({
        'U': [(1,0,1), (0,0,-1)],
        'V': [(0,0,1), (0,1,1)],
        'W': [(1,1,1)]
    })
    
    # M7: (A12-A22)(B21+B22)
    # A: A12 (0,1) coeff 1, A22 (1,1) coeff -1
    # B: B21 (1,0), B22 (1,1)
    # C: C11 (0,0) coeff 1
    strassen_terms.append({
        'U': [(0,1,1), (1,1,-1)],
        'V': [(1,0,1), (1,1,1)],
        'W': [(0,0,1)]
    })
    
    # Define the 6 instances
    # Instance (a_block, b_block, c_block)
    # a_block: row offset for A (0 or 2), col offset for A (0) -> A is 2x4, split cols 0-1 and 2-3
    # Actually, A is 2x4. Rows 0-1. Cols 0-3.
    # Split A into A_L (cols 0-1) and A_R (cols 2-3).
    # B is 4x6. Rows 0-3. Cols 0-5.
    # Split B into B_T (rows 0-1) and B_B (rows 2-3).
    # C is 2x6. Rows 0-1. Cols 0-5.
    
    # We have 3 splits of the 6 columns of B/C into 2s.
    # k_offsets: [0, 2, 4]
    
    instances = []
    # A_L (j_offset 0), B_T (j_offset 0)
    for k_off in [0, 2, 4]:
        instances.append({
            'a_row_off': 0, 'a_col_off': 0, # A is 2x2 block at (0,0)
            'b_row_off': 0, 'b_col_off': k_off, # B is 2x2 block at (0, k_off)
            'c_row_off': k_off, 'c_col_off': 0 # C is 2x2 block at (k_off, 0)
        })
    # A_R (j_offset 2), B_B (j_offset 2)
    for k_off in [0, 2, 4]:
        instances.append({
            'a_row_off': 0, 'a_col_off': 2, # A is 2x2 block at (0,2)
            'b_row_off': 2, 'b_col_off': k_off, # B is 2x2 block at (2, k_off)
            'c_row_off': k_off, 'c_col_off': 0 # C is 2x2 block at (k_off, 0)
        })
        
    col = 0
    for inst in instances:
        if time.time() > deadline:
            break
            
        for term in strassen_terms:
            if time.time() > deadline:
                break
                
            # Fill U
            for u_r, u_c, coeff in term['U']:
                # Global A index: row = a_row_off + u_r, col = a_col_off + u_c
                # U index: row * m + col
                u_idx = (inst['a_row_off'] + u_r) * m + (inst['a_col_off'] + u_c)
                U[u_idx, col] += coeff
                
            # Fill V
            for v_r, v_c, coeff in term['V']:
                # Global B index: row = b_row_off + v_r, col = b_col_off + v_c
                # V index: row * p + col
                v_idx = (inst['b_row_off'] + v_r) * p + (inst['b_col_off'] + v_c)
                V[v_idx, col] += coeff
                
            # Fill W
            for w_r, w_c, coeff in term['W']:
                # Global C index: row = c_row_off + w_r, col = c_col_off + w_c
                # W index: row * n + col (Trace convention W[k*n+i] where k is row of C, i is col of C)
                w_idx = (inst['c_row_off'] + w_r) * n + (inst['c_col_off'] + w_c)
                W[w_idx, col] += coeff
                
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
