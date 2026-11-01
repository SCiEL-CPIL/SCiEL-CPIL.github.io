"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x10 matrix multiplication.
    Uses block-diagonal Strassen (rank 7 per block, 5 blocks -> rank 35).
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Strassen coefficients for 2x2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    u_coeffs = [
        [1, 0, 0, 1],  # P1: a+d
        [0, 0, 1, 1],  # P2: c+d
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: a+b
        [-1, 0, 1, 0], # P6: c-a
        [0, 1, 0, -1]  # P7: b-d
    ]
    
    # For each block b in 0..4:
    # B indices: e=2b, f=2b+1, g=10+2b, h=10+2b+1
    # C indices: c00=4b, c01=4b+2, c10=4b+1, c11=4b+3
    # (Note: C is 2x10, so C[k*n+i] with n=2 -> C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3)
    
    # V coefficients for each Strassen product (relative to e,f,g,h)
    # P1: e+h -> [1,0,0,1]
    # P2: e    -> [1,0,0,0]
    # P3: f-h  -> [0,1,0,-1]
    # P4: g-e  -> [0,0,1,-1]  (Wait, standard Strassen P4 is d, P3 is a. Let's stick to the current working mapping)
    
    # Let's reuse the exact mapping from the known valid solution to ensure correctness.
    # The previous solution used:
    # P1: V[e]=1, V[h]=1
    # P2: V[e]=1
    # P3: V[f]=1, V[h]=-1
    # P4: V[g]=1, V[e]=-1
    # P5: V[h]=1
    # P6: V[e]=1, V[f]=1
    # P7: V[g]=1, V[h]=1
    
    # And W:
    # P1: W[c00]=1, W[c11]=1
    # P2: W[c10]=1, W[c11]=-1
    # P3: W[c01]=1, W[c11]=1
    # P4: W[c00]=1, W[c10]=1
    # P5: W[c00]=-1, W[c01]=1
    # P6: W[c11]=1
    # P7: W[c00]=1
    
    # This specific set of coefficients corresponds to a valid Strassen-like decomposition.
    # We will replicate this structure.
    
    for b in range(5):
        for s in range(7):
            col = b * 7 + s
            
            # Fill U (same for all blocks)
            for i in range(4):
                U[i, col] = u_coeffs[s][i]
                
            # Indices for B
            e = 2 * b
            f = 2 * b + 1
            g = 10 + 2 * b
            h = 10 + 2 * b + 1
            
            # Fill V
            if s == 0:
                V[e, col] = 1
                V[h, col] = 1
            elif s == 1:
                V[e, col] = 1
            elif s == 2:
                V[f, col] = 1
                V[h, col] = -1
            elif s == 3:
                V[g, col] = 1
                V[e, col] = -1
            elif s == 4:
                V[h, col] = 1
            elif s == 5:
                V[e, col] = 1
                V[f, col] = 1
            elif s == 6:
                V[g, col] = 1
                V[h, col] = 1
                
            # Indices for C
            c00 = 4 * b
            c01 = 4 * b + 2
            c10 = 4 * b + 1
            c11 = 4 * b + 3
            
            # Fill W
            if s == 0:
                W[c00, col] = 1
                W[c11, col] = 1
            elif s == 1:
                W[c10, col] = 1
                W[c11, col] = -1
            elif s == 2:
                W[c01, col] = 1
                W[c11, col] = 1
            elif s == 3:
                W[c00, col] = 1
                W[c10, col] = 1
            elif s == 4:
                W[c00, col] = -1
                W[c01, col] = 1
            elif s == 5:
                W[c11, col] = 1
            elif s == 6:
                W[c00, col] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
