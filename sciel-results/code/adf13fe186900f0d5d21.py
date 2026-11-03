import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: j in {0, 1}
    # This is M_{2, 2, 8}
    # Split k into 4 groups of 2
    for j_part in range(2): # 0 for j=0,1 ; 1 for j=2,3
        j_offset = j_part * 2
        for k_group in range(4): # 4 groups of 2 k's
            k_offset = k_group * 2
            # Strassen 2x2x2
            # i in {0,1}, j_local in {0,1}, k_local in {0,1}
            
            # M1 = (A_00 + A_11) * (B_00 + B_11)
            U[j_offset*2 + 0][col] = 1 # A_00 (i=0, j=j_offset)
            U[j_offset*2 + 1][col] = 1 # A_11 (i=1, j=j_offset+1)
            V[j_offset*2 + 0][col] = 1 # B_00 (j=j_offset, k=k_offset)
            V[j_offset*2 + 1][col] = 1 # B_11 (j=j_offset+1, k=k_offset+1)
            W[k_offset*2 + 0][col] = 1 # C_00 (k=k_offset, i=0)
            W[k_offset*2 + 1][col] = 1 # C_11 (k=k_offset+1, i=1)
            col += 1
            
            # M2 = (A_10 + A_11) * B_00
            U[j_offset*2 + 1][col] = 1 # A_10 (i=1, j=j_offset)
            U[j_offset*2 + 1][col] = 1 # A_11 (i=1, j=j_offset+1)
            V[j_offset*2 + 0][col] = 1 # B_00 (j=j_offset, k=k_offset)
            W[k_offset*2 + 1][col] = 1 # C_10 (k=k_offset, i=1)
            col += 1
            
            # M3 = A_00 * (B_10 - B_11)
            U[j_offset*2 + 0][col] = 1 # A_00 (i=0, j=j_offset)
            V[j_offset*2 + 1][col] = 1 # B_10 (j=j_offset+1, k=k_offset)
            V[j_offset*2 + 1][col] = -1 # B_11 (j=j_offset+1, k=k_offset+1)
            W[k_offset*2 + 0][col] = 1 # C_01 (k=k_offset+1, i=0)
            col += 1
            
            # M4 = A_11 * (B_01 - B_00)
            U[j_offset*2 + 1][col] = 1 # A_11 (i=1, j=j_offset+1)
            V[j_offset*2 + 0][col] = 1 # B_01 (j=j_offset, k=k_offset+1)
            V[j_offset*2 + 0][col] = -1 # B_00 (j=j_offset, k=k_offset)
            W[k_offset*2 + 1][col] = 1 # C_10 (k=k_offset, i=1)
            W[k_offset*2 + 0][col] = 1 # C_00 (k=k_offset, i=0)
            col += 1
            
            # M5 = (A_00 + A_01) * B_11
            U[j_offset*2 + 0][col] = 1 # A_00 (i=0, j=j_offset)
            U[j_offset*2 + 0][col] = 1 # A_01 (i=0, j=j_offset+1)
            V[j_offset*2 + 1][col] = 1 # B_11 (j=j_offset+1, k=k_offset+1)
            W[k_offset*2 + 0][col] = -1 # C_00 (k=k_offset, i=0)
            W[k_offset*2 + 1][col] = 1 # C_01 (k=k_offset+1, i=0)
            col += 1
            
            # M6 = (A_10 - A_00) * (B_00 + B_01)
            U[j_offset*2 + 1][col] = 1 # A_10 (i=1, j=j_offset)
            U[j_offset*2 + 0][col] = -1 # A_00 (i=0, j=j_offset)
            V[j_offset*2 + 0][col] = 1 # B_00 (j=j_offset, k=k_offset)
            V[j_offset*2 + 0][col] = 1 # B_01 (j=j_offset, k=k_offset+1)
            W[k_offset*2 + 1][col] = 1 # C_11 (k=k_offset+1, i=1)
            col += 1
            
            # M7 = (A_01 - A_11) * (B_10 + B_11)
            U[j_offset*2 + 0][col] = 1 # A_01 (i=0, j=j_offset+1)
            U[j_offset*2 + 1][col] = -1 # A_11 (i=1, j=j_offset+1)
            V[j_offset*2 + 1][col] = 1 # B_10 (j=j_offset+1, k=k_offset)
            V[j_offset*2 + 1][col] = 1 # B_11 (j=j_offset+1, k=k_offset+1)
            W[k_offset*2 + 0][col] = 1 # C_00 (k=k_offset, i=0)
            W[k_offset*2 + 1][col] = 1 # C_11 (k=k_offset+1, i=1)
            col += 1
            
    return {"U": U, "V": V, "W": W}
