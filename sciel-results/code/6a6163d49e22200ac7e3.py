"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define the Strassen blocks
    # i in {0,1,2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    # Blocks are defined by (i_start, j_start, k_start)
    # We can fit:
    # i: 0-1, 2-3
    # j: 0-1 (only one 2x2 block fits in 5)
    # k: 0-1, 2-3, 4-5
    strassen_blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (2, 0, 0),
        (2, 0, 2),
        (2, 0, 4)
    ]
    
    # Create a set of (i,j,k) tuples that are covered by Strassen blocks
    covered = set()
    for i0, j0, k0 in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
                    
    # Initialize U, V, W
    # We don't know the exact rank yet, so we build lists and then convert
    # Rank = (Total - Covered) + 7 * NumBlocks
    total_terms = n * m * p
    num_covered = len(covered)
    num_blocks = len(strassen_blocks)
    rank = (total_terms - num_covered) + 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Add Schoolbook terms for non-covered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    # 2. Add Strassen terms for each block
    # Strassen formulas for 2x2 matrices:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # u_coefs: list of 7 dicts {(di,dj): coeff}
    # v_coefs: list of 7 dicts {(dj,dk): coeff}
    # w_coefs: list of 7 dicts {(di,dk): coeff}
    
    u_coefs = [
        {(0,0): 1, (1,1): 1}, # P1
        {(1,0): 1, (1,1): 1}, # P2
        {(0,0): 1},           # P3
        {(1,1): 1},           # P4
        {(0,0): 1, (0,1): 1}, # P5
        {(1,0): 1, (0,0): -1},# P6
        {(0,1): 1, (1,1): -1} # P7
    ]
    
    v_coefs = [
        {(0,0): 1, (1,1): 1}, # P1
        {(0,0): 1},           # P2
        {(0,1): 1, (1,1): -1},# P3
        {(1,0): 1, (0,0): -1},# P4
        {(1,1): 1},           # P5
        {(0,0): 1, (0,1): 1}, # P6
        {(1,0): 1, (1,1): 1}  # P7
    ]
    
    w_coefs = [
        {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): 1}, # P1 -> C00(+1), C11(+1), C01(-1? No, C01=P3+P5), C10(+1? No)
        # Let's map carefully:
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # P1: C00(+1), C11(+1)
        # P2: C10(+1), C11(-1)
        # P3: C01(+1), C11(+1)
        # P4: C00(+1), C10(+1)
        # P5: C00(-1), C01(+1)
        # P6: C11(+1)
        # P7: C00(+1)
        
        {(0,0): 1, (1,1): 1}, # P1
        {(1,0): 1, (1,1): -1},# P2
        {(0,1): 1, (1,1): 1}, # P3
        {(0,0): 1, (1,0): 1}, # P4
        {(0,0): -1, (0,1): 1},# P5
        {(1,1): 1},           # P6
        {(0,0): 1}            # P7
    ]
    
    for i0, j0, k0 in strassen_blocks:
        for m_idx in range(7):
            # Fill U
            for (di, dj), coef in u_coefs[m_idx].items():
                i = i0 + di
                j = j0 + dj
                U[i * m + j][col] = coef
                
            # Fill V
            for (dj, dk), coef in v_coefs[m_idx].items():
                j = j0 + dj
                k = k0 + dk
                V[j * p + k][col] = coef
                
            # Fill W
            for (di, dk), coef in w_coefs[m_idx].items():
                i = i0 + di
                k = k0 + dk
                W[k * n + i][col] = coef
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
