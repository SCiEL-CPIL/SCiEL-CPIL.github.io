"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Use the Kronecker product of two Strassen decompositions.
    # Rank = 7 * 7 = 49.
    
    # Strassen decomposition for 2x2 matrix multiplication.
    # Indices 0,1,2,3 correspond to a,b,c,d for A and e,f,g,h for B.
    # W index is k*n+i, so for n=2, k*c+a.
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 1]
    ], dtype=float)
    
    n, m, p = DIMENSIONS
    # n=m=p=4
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # Mapping from standard 4x4 index (i*4+j) to Kronecker indices
    # i = ib*2 + ii, j = jb*2 + ji
    # Std Index = ib*8 + ii*4 + jb*2 + ji
    # Kron Index for U/V: I_out*4 + I_in
    # I_out = ib*2 + jb (for A/B)
    # I_in = ii*2 + ji (for A/B)
    
    # For W:
    # Std Index = k*4 + i = kb*8 + ki*4 + ib*2 + ii
    # I_out_W = kb*2 + ib
    # I_in_W = ki*2 + ii
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            for idx in range(16):
                # Decompose idx for U and V (A and B matrices)
                ib = idx // 8
                ii = (idx // 4) % 2
                jb = (idx // 2) % 2
                ji = idx % 2
                
                I_out_UV = ib * 2 + jb
                I_in_UV = ii * 2 + ji
                Kron_UV = I_out_UV * 4 + I_in_UV
                
                U[idx, r] = U2[I_out_UV, r1] * U2[I_in_UV, r2]
                V[idx, r] = V2[I_out_UV, r1] * V2[I_in_UV, r2]
                
                # Decompose idx for W (C matrix)
                # idx = k*4 + i
                kb = idx // 8
                ki = (idx // 4) % 2
                ib_w = (idx // 2) % 2
                ii_w = idx % 2
                
                I_out_W = kb * 2 + ib_w
                I_in_W = ki * 2 + ii_w
                Kron_W = I_out_W * 4 + I_in_W
                
                W[idx, r] = W2[I_out_W, r1] * W2[I_in_W, r2]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
