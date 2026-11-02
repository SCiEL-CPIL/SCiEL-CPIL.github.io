"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * (7 + 2) = 36
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Iterate over the 4 blocks of B (3x2 each)
    for b in range(4):
        # k indices for this block: 2b, 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Strassen part for A1 (2x2) * B1 (2x2) ---
        # A1 uses A[0,0], A[0,1], A[1,0], A[1,1]
        # B1 uses B[0,k0], B[0,k1], B[1,k0], B[1,k1]
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0 * m + 0][col] = 1  # A00
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k0][col] = 1 # B0k0
        V[1 * p + k1][col] = 1 # B1k1
        # W contributions: C0k0 += 1, C1k1 += 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[1 * m + 0][col] = 1  # A10
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k0][col] = 1 # B0k0
        # W contributions: C1k0 += 1, C1k1 += -1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0 * m + 0][col] = 1  # A00
        V[0 * p + k1][col] = 1 # B0k1
        V[1 * p + k1][col] = -1 # B1k1
        # W contributions: C0k1 += 1, C1k1 += 1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[1 * m + 1][col] = 1  # A11
        V[1 * p + k0][col] = 1 # B1k0
        V[0 * p + k0][col] = -1 # B0k0
        # W contributions: C0k0 += 1, C1k0 += 1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0 * m + 0][col] = 1  # A00
        U[0 * m + 1][col] = 1  # A01
        V[1 * p + k1][col] = 1 # B1k1
        # W contributions: C0k0 += -1, C0k1 += 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[1 * m + 0][col] = 1  # A10
        U[0 * m + 0][col] = -1 # A00
        V[0 * p + k0][col] = 1 # B0k0
        V[0 * p + k1][col] = 1 # B0k1
        # W contributions: C1k1 += 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[0 * m + 1][col] = 1  # A01
        U[1 * m + 1][col] = -1 # A11
        V[1 * p + k0][col] = 1 # B1k0
        V[1 * p + k1][col] = 1 # B1k1
        # W contributions: C0k0 += 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # --- Rank-2 part for A2 (2x1) * B2 (1x2) ---
        # A2 uses A[0,2], A[1,2]
        # B2 uses B[2,k0], B[2,k1]
        
        # P1 = (A02 + A12) * (B2k0 + B2k1)
        U[0 * m + 2][col] = 1  # A02
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k0][col] = 1 # B2k0
        V[2 * p + k1][col] = 1 # B2k1
        # W contributions:
        # C0k0 += 1/4, C0k1 += 1/4
        # C1k0 += -1/4, C1k1 += -1/4
        W[k0 * n + 0][col] = 0.25
        W[k1 * n + 0][col] = 0.25
        W[k0 * n + 1][col] = -0.25
        W[k1 * n + 1][col] = -0.25
        col += 1
        
        # P2 = (A02 - A12) * (B2k0 - B2k1)
        U[0 * m + 2][col] = 1  # A02
        U[1 * m + 2][col] = -1 # A12
        V[2 * p + k0][col] = 1 # B2k0
        V[2 * p + k1][col] = -1 # B2k1
        # W contributions:
        # C0k0 += 1/4, C0k1 += -1/4
        # C1k0 += 1/4, C1k1 += -1/4
        W[k0 * n + 0][col] = 0.25
        W[k1 * n + 0][col] = -0.25
        W[k0 * n + 1][col] = 0.25
        W[k1 * n + 1][col] = -0.25
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
