"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    # We use 7 Strassen blocks (rank 7 each) for the first 14 columns
    # and 1 Schoolbook block (rank 4) for the last column.
    # Total rank = 7 * 7 + 4 = 53.
    
    rank = 53
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    
    col_offset = 0
    
    # Define Strassen coefficients for a 2x2x2 block
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (relative to block start)
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (relative to block start)
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # M3 = A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),
        # M4 = A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: -1})
    ]
    
    # Process 7 Strassen blocks
    for block_idx in range(7):
        b_start = block_idx * 2
        # B indices in global V:
        # B_{0, b_start} -> V[0 * p + b_start]
        # B_{0, b_start+1} -> V[0 * p + b_start + 1]
        # B_{1, b_start} -> V[1 * p + b_start]
        # B_{1, b_start+1} -> V[1 * p + b_start + 1]
        
        # C indices in global W:
        # C_{0, b_start} -> W[b_start * n + 0]
        # C_{0, b_start+1} -> W[(b_start+1) * n + 0]
        # C_{1, b_start} -> W[b_start * n + 1]
        # C_{1, b_start+1} -> W[(b_start+1) * n + 1]
        
        for term_idx in range(7):
            r = col_offset
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[term_idx]
            
            # Fill U (A part)
            for a_idx, coeff in a_coeffs.items():
                U[a_idx][r] = coeff
                
            # Fill V (B part)
            # Local B index 0 -> 00, 1 -> 01, 2 -> 10, 3 -> 11
            for b_idx, coeff in b_coeffs.items():
                if b_idx == 0: # 00
                    v_idx = 0 * p + b_start
                elif b_idx == 1: # 01
                    v_idx = 0 * p + b_start + 1
                elif b_idx == 2: # 10
                    v_idx = 1 * p + b_start
                elif b_idx == 3: # 11
                    v_idx = 1 * p + b_start + 1
                V[v_idx][r] = coeff
                
            # Fill W (C part)
            # Local C index 0 -> 00, 1 -> 01, 2 -> 10, 3 -> 11
            for c_idx, coeff in c_coeffs.items():
                if c_idx == 0: # 00
                    w_idx = b_start * n + 0
                elif c_idx == 1: # 01
                    w_idx = (b_start + 1) * n + 0
                elif c_idx == 2: # 10
                    w_idx = b_start * n + 1
                elif c_idx == 3: # 11
                    w_idx = (b_start + 1) * n + 1
                W[w_idx][r] = coeff
                
            col_offset += 1
            
    # Process 1 Schoolbook block for the last column (index 14)
    # C_{0, 14} = A_{00} B_{0, 14} + A_{01} B_{1, 14}
    # C_{1, 14} = A_{10} B_{0, 14} + A_{11} B_{1, 14}
    
    b_last = 14
    schoolbook_terms = [
        # (A_idx, B_idx, C_idx)
        (0, 0 * p + b_last, b_last * n + 0), # A00 * B0,14 -> C0,14
        (1, 1 * p + b_last, b_last * n + 0), # A01 * B1,14 -> C0,14
        (2, 0 * p + b_last, b_last * n + 1), # A10 * B0,14 -> C1,14
        (3, 1 * p + b_last, b_last * n + 1)  # A11 * B1,14 -> C1,14
    ]
    
    for a_idx, b_idx, c_idx in schoolbook_terms:
        r = col_offset
        U[a_idx][r] = 1
        V[b_idx][r] = 1
        W[c_idx][r] = 1
        col_offset += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
