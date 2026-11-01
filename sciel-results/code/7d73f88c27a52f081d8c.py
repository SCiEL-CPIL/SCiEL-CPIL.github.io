"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    
    # Initialize U, V, W
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen rank-7 decomposition for 2x2 matrix multiplication
    # T = sum_c u_c v_c w_c
    # Indices for A (2x2): 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    # Indices for B (2x2): 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    # Indices for C (2x2): 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    
    # The 7 terms for M_{2,2,2}
    # u: coeffs for A, v: coeffs for B, w: coeffs for C
    # Note: w is the same for all terms in a given Strassen block, but we need to map to global C indices.
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B10-B11) -> C01 += M3
    # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # Let's define the u, v, w vectors for each of the 7 terms.
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    terms = []
    
    # M1
    u1 = [1, 0, 0, 1] # A00 + A11
    v1 = [1, 0, 0, 1] # B00 + B11
    w1 = [1, 0, 0, 1] # C00 + C11
    terms.append((u1, v1, w1))
    
    # M2
    u2 = [0, 0, 1, 1] # A10 + A11
    v2 = [1, 0, 0, 0] # B00
    w2 = [0, 0, 1, 0] # C10
    terms.append((u2, v2, w2))
    
    # M3
    u3 = [1, 0, 0, 0] # A00
    v3 = [0, -1, 1, 0] # B10 - B11
    w3 = [0, 1, 0, 0] # C01
    terms.append((u3, v3, w3))
    
    # M4
    u4 = [0, 0, 0, 1] # A11
    v4 = [-1, 1, 0, 0] # B01 - B00
    w4 = [1, 0, 1, 0] # C00 + C10
    terms.append((u4, v4, w4))
    
    # M5
    u5 = [1, 1, 0, 0] # A00 + A01
    v5 = [0, 0, 0, 1] # B11
    w5 = [-1, 1, 0, 0] # -C00 + C01
    terms.append((u5, v5, w5))
    
    # M6
    u6 = [-1, 0, 1, 0] # A10 - A00
    v6 = [1, 1, 0, 0] # B00 + B01
    w6 = [0, 0, 0, 1] # C11
    terms.append((u6, v6, w6))
    
    # M7
    u7 = [0, 1, 0, -1] # A01 - A11
    v7 = [0, 0, 1, 1] # B10 + B11
    w7 = [1, 0, 0, 0] # C00
    terms.append((u7, v7, w7))
    
    # Now map these to the 2x4 by 4x8 problem.
    # A is 2x4. Indices (i, j) with i in 0..1, j in 0..3.
    # Global A index: i*4 + j.
    # B is 4x8. Indices (j, k) with j in 0..3, k in 0..7.
    # Global B index: j*8 + k.
    # C is 2x8. Indices (i, k) with i in 0..1, k in 0..7.
    # Global C index: i*8 + k.
    
    # We split A into A0 (cols 0,1) and A1 (cols 2,3).
    # We split B into B0 (rows 0,1) and B1 (rows 2,3).
    # We split C into C0 (cols 0..3) and C1 (cols 4..7).
    
    # For each block l in 0..3 (corresponding to B cols l*2, l*2+1):
    #   We compute C_block = A0 * B0_block + A1 * B1_block.
    #   B0_block is rows 0,1 of B, cols l*2, l*2+1.
    #   B1_block is rows 2,3 of B, cols l*2, l*2+1.
    
    col = 0
    for l in range(4):
        if time.time() > deadline:
            break
            
        # Part 1: A0 * B0_block
        # A0 is A[:, 0:2]. Global indices: i*4 + 0, i*4 + 1.
        # B0_block is B[0:2, l*2:l*2+2]. Global indices: 0*8 + l*2 + dk, 1*8 + l*2 + dk.
        # C0_block is C[:, l*2:l*2+2]. Global indices: i*8 + l*2 + dk.
        
        for u, v, w in terms:
            # Add term to U, V, W
            # U: A0 indices
            for i in range(2):
                for j_local in range(2):
                    val = u[i*2 + j_local]
                    if val != 0:
                        # Global A index: i*4 + j_local
                        U[i*4 + j_local][col] += val
                        
            # V: B0_block indices
            for j_local in range(2):
                for k_local in range(2):
                    val = v[j_local*2 + k_local]
                    if val != 0:
                        # Global B index: j_local*8 + (l*2 + k_local)
                        V[j_local*8 + l*2 + k_local][col] += val
                        
            # W: C0_block indices
            for i in range(2):
                for k_local in range(2):
                    val = w[i*2 + k_local]
                    if val != 0:
                        # Global C index: i*8 + (l*2 + k_local)
                        W[i*8 + l*2 + k_local][col] += val
                        
            col += 1
            
        # Part 2: A1 * B1_block
        # A1 is A[:, 2:4]. Global indices: i*4 + 2, i*4 + 3.
        # B1_block is B[2:4, l*2:l*2+2]. Global indices: 2*8 + l*2 + dk, 3*8 + l*2 + dk.
        # C1_block is C[:, l*2:l*2+2]. Global indices: i*8 + l*2 + dk.
        
        for u, v, w in terms:
            # Add term to U, V, W
            # U: A1 indices
            for i in range(2):
                for j_local in range(2):
                    val = u[i*2 + j_local]
                    if val != 0:
                        # Global A index: i*4 + (2 + j_local)
                        U[i*4 + 2 + j_local][col] += val
                        
            # V: B1_block indices
            for j_local in range(2):
                for k_local in range(2):
                    val = v[j_local*2 + k_local]
                    if val != 0:
                        # Global B index: (2 + j_local)*8 + (l*2 + k_local)
                        V[(2 + j_local)*8 + l*2 + k_local][col] += val
                        
            # W: C1_block indices (same C indices as Part 1)
            for i in range(2):
                for k_local in range(2):
                    val = w[i*2 + k_local]
                    if val != 0:
                        # Global C index: i*8 + (l*2 + k_local)
                        W[i*8 + l*2 + k_local][col] += val
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
