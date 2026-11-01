"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen decomposition for 2x2 blocks of B.
    # Split 13 columns into six 2-column blocks and one 1-column block.
    # Rank = 6*7 + 4 = 46
    
    rank = 6 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Process 6 blocks of 2 columns
    for c in range(0, 12, 2):
        # Strassen 7 terms for 2x2 * 2x2
        # M1 = (A00 + A11) * (B0c + B1,c+1)
        # M2 = (A10 + A11) * B0c
        # M3 = A00 * B1,c+1
        # M4 = A11 * B0c
        # M5 = (A00 + A01) * B1,c+1
        # M6 = (A10 - A00) * (B0c + B1c)
        # M7 = (A01 - A11) * (B1c + B1,c+1)
        
        # C0c = M1 - M4 + M5 - M7
        # C0,c+1 = M3 + M7
        # C1c = M2 + M4 - M6 - M7
        # C1,c+1 = M1 + M3 - M5 + M6
        
        # M1
        U[0][col] = 1  # A00
        U[3][col] = 1  # A11
        V[c][col] = 1  # B0c
        V[13 + c + 1][col] = 1  # B1,c+1
        W[2*c][col] = 1  # C0c
        W[2*c+1][col] = -1  # C1c
        W[2*c+3][col] = 1  # C1,c+1
        col += 1
        
        # M2
        U[2][col] = 1  # A10
        U[3][col] = 1  # A11
        V[c][col] = 1  # B0c
        W[2*c+1][col] = 1  # C1c
        col += 1
        
        # M3
        U[0][col] = 1  # A00
        V[13 + c + 1][col] = 1  # B1,c+1
        W[2*c+2][col] = 1  # C0,c+1
        W[2*c+3][col] = 1  # C1,c+1
        col += 1
        
        # M4
        U[3][col] = 1  # A11
        V[c][col] = 1  # B0c
        W[2*c][col] = -1  # C0c
        W[2*c+1][col] = 1  # C1c
        col += 1
        
        # M5
        U[0][col] = 1  # A00
        U[1][col] = 1  # A01
        V[13 + c + 1][col] = 1  # B1,c+1
        W[2*c][col] = 1  # C0c
        W[2*c+3][col] = -1  # C1,c+1
        col += 1
        
        # M6
        U[2][col] = 1  # A10
        U[0][col] = -1  # A00
        V[c][col] = 1  # B0c
        V[13 + c][col] = 1  # B1c
        W[2*c+1][col] = -1  # C1c
        W[2*c+3][col] = 1  # C1,c+1
        col += 1
        
        # M7
        U[1][col] = 1  # A01
        U[3][col] = -1  # A11
        V[13 + c][col] = 1  # B1c
        V[13 + c + 1][col] = 1  # B1,c+1
        W[2*c][col] = -1  # C0c
        W[2*c+2][col] = 1  # C0,c+1
        W[2*c+1][col] = -1  # C1c
        col += 1
    
    # Last 1-column block (c=12): schoolbook 2x2 * 2x1
    # C0,12 = A00*B0,12 + A01*B1,12
    # C1,12 = A10*B0,12 + A11*B1,12
    
    # Term 1: A00 * B0,12
    U[0][col] = 1
    V[12][col] = 1
    W[24][col] = 1  # C0,12 -> 12*2+0 = 24
    col += 1
    
    # Term 2: A01 * B1,12
    U[1][col] = 1
    V[25][col] = 1  # B1,12 -> 13+12 = 25
    W[24][col] = 1
    col += 1
    
    # Term 3: A10 * B0,12
    U[2][col] = 1
    V[12][col] = 1
    W[25][col] = 1  # C1,12 -> 12*2+1 = 25
    col += 1
    
    # Term 4: A11 * B1,12
    U[3][col] = 1
    V[25][col] = 1
    W[25][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
