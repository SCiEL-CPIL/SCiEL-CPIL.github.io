import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28
    rank = 28
    
    # Initialize matrices
    # U: (n*m, rank) -> (8, 28)
    # V: (m*p, rank) -> (16, 28)
    # W: (p*n, rank) -> (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[i, j], [k, l]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # i = M1 + M4 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M3 + M6
    
    # We need to generate the rank-1 terms for a 2x2x2 multiplication.
    # For each term r, we have:
    # U_r (2x2 flattened to 4? No, U is for A, V for B, W for C?
    # Wait. The tensor is C = A B.
    # Decomposition: C_{ik} = sum_r U_{ik}^{(r)} V_{ij}^{(r)} W_{jk}^{(r)}?
    # No.
    # Standard: C_{ik} = sum_r alpha_{ik}^{(r)} beta_{ij}^{(r)} gamma_{jk}^{(r)}
    # Here beta is on A (indices i,j), gamma is on B (indices j,k), alpha is on C (indices i,k).
    
    # In our global problem:
    # U corresponds to A (indices i,j).
    # V corresponds to B (indices j,k).
    # W corresponds to C (indices i,k).
    
    # So for a 2x2x2 block:
    # U_block is 2x2 (flattened to 4 entries in global U, but we only fill the relevant 4 entries)
    # V_block is 2x2 (flattened to 4 entries in global V)
    # W_block is 2x2 (flattened to 4 entries in global W)
    
    # Let's define the 7 Strassen terms.
    # Each term is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(i,j): val}
    # V_coeffs: dict {(j,k): val}
    # W_coeffs: dict {(i,k): val}
    
    def get_strassen_terms():
        terms = []
        # M1
        terms.append({
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
            'W': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}
        })
        # M2
        terms.append({
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0},
            'W': {(1,0): 1, (1,1): -1, (0,0): 0, (0,1): 0}
        })
        # M3
        terms.append({
            'U': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0},
            'V': {(0,1): 1, (0,0): 0, (1,1): -1, (1,0): 0},
            'W': {(0,1): 1, (1,1): 0, (0,0): 0, (1,0): 0}
        })
        # M4
        terms.append({
            'U': {(1,1): 1, (1,0): 0, (0,0): 0, (0,1): 0},
            'V': {(1,0): 1, (1,1): 0, (0,0): -1, (0,1): 0},
            'W': {(0,0): 1, (1,0): 1, (0,1): 0, (1,1): 0}
        })
        # M5
        terms.append({
            'U': {(0,0): 1, (0,1): 1, (1,0): 0, (1,1): 0},
            'V': {(1,1): 1, (1,0): 0, (0,0): 0, (0,1): 0},
            'W': {(0,0): -1, (1,0): 0, (0,1): 1, (1,1): 0}
        })
        # M6
        terms.append({
            'U': {(1,0): 1, (0,0): -1, (1,1): 0, (0,1): 0},
            'V': {(0,0): 1, (0,1): 1, (1,0): 0, (1,1): 0},
            'W': {(1,1): 1, (0,1): 0, (1,0): 0, (0,0): 0}
        })
        # M7
        terms.append({
            'U': {(0,1): 1, (1,1): -1, (0,0): 0, (1,0): 0},
            'V': {(1,0): 1, (1,1): 1, (0,0): 0, (0,1): 0},
            'W': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}
        })
        return terms

    strassen_terms = get_strassen_terms()
    
    # Function to add a block decomposition to the global matrices
    # a_offset: row offset in A (0 or 2? No, A is 2x4. Rows are 0,1. Cols are 0-3)
    # For Left part: A cols 0,1. B rows 0,1.
    # For Right part: A cols 2,3. B rows 2,3.
    
    def add_block(a_col_offset, b_row_offset, base_col_offset):
        # a_col_offset: 0 for left, 2 for right
        # b_row_offset: 0 for top, 2 for bottom
        # base_col_offset: 0 for B_1 (cols 0,1 of B_part), 2 for B_2 (cols 2,3 of B_part)
        
        for t_idx in range(7):
            term = strassen_terms[t_idx]
            global_col = t_idx + (base_col_offset * 7) # 0-6 or 7-13
            
            # Fill U (A part)
            for (i, j), val in term['U'].items():
                # Global A index: i * m + (a_col_offset + j)
                global_i = i
                global_j = a_col_offset + j
                u_idx = global_i * m + global_j
                U[u_idx][global_col] = val
                
            # Fill V (B part)
            for (j, k), val in term['V'].items():
                # Global B index: (b_row_offset + j) * p + k
                global_j = b_row_offset + j
                global_k = k
                v_idx = global_j * p + global_k
                V[v_idx][global_col] = val
                
            # Fill W (C part)
            for (i, k), val in term['W'].items():
                # Global C index: k * n + i
                global_i = i
                global_k = k
                w_idx = global_k * n + global_i
                W[w_idx][global_col] = val

    # Left part: A cols 0,1. B rows 0,1.
    # B is split into B_1 (cols 0,1) and B_2 (cols 2,3)
    # Wait. In the sub-problem A_L * B_T, B_T is 2x4.
    # We split B_T into two 2x2 blocks: B_T_1 (cols 0,1) and B_T_2 (cols 2,3).
    # So base_col_offset for B_1 is 0, for B_2 is 2.
    
    # Left part (A cols 0,1, B rows 0,1)
    add_block(a_col_offset=0, b_row_offset=0, base_col_offset=0) # B_T_1
    add_block(a_col_offset=0, b_row_offset=0, base_col_offset=2) # B_T_2
    
    # Right part (A cols 2,3, B rows 2,3)
    add_block(a_col_offset=2, b_row_offset=2, base_col_offset=0) # B_B_1
    add_block(a_col_offset=2, b_row_offset=2, base_col_offset=2) # B_B_2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
