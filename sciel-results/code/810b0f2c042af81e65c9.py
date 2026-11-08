import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 4, 8
    
    # Use a refined block-Strassen decomposition for 2x4 * 4x8
    # Partition: A = [A0 | A1] (each 2x2), B = [B0; B1] (each 4x4)
    # C = A0 @ B0 + A1 @ B1 (2x8 result)
    # For each 2x2 @ 4x4, we can use a rank-7 per 2x2 sub-block approach
    # But we can do better by sharing across the 4x4 B blocks
    
    # Alternative: treat as 2x(2x2) * (2x2)x(2x4) = use Kronecker structure
    # A = a0 ⊗ I2 + a1 ⊗ S2 where a0, a1 are 1x2, S2 = [[0,1],[1,0]]
    # B = b0 ⊗ T4 + b1 ⊗ U4 where b0, b1 are 2x2, T4, U4 are 2x4
    
    # Simpler: use direct rank-52 construction via careful combination
    # We'll use the same 7-term Strassen per 2x2 block but combine terms
    
    x = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=int)
    
    y = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=int)
    
    z = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Build rank-56 decomposition with explicit indexing
    rank = 56
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    r = 0
    for i_block in range(2):  # A row block
        for j_block in range(2):  # A col block (2x2 sub-blocks)
            for k_block in range(2):  # B row block
                for s in range(7):  # Strassen term
                    for i in range(2):  # row within 2x2
                        for j in range(2):  # col within 2x2
                            U[i*m + i_block*2 + j, r] = x[s, i*2 + j]
                    for j in range(2):
                        for k in range(2):
                            V[(i_block*2+j)*p + k_block*2 + k, r] = y[s, j*2 + k]
                    for i in range(2):
                        for k in range(4):
                            if k < 2*k_block + 2 and k >= 2*k_block:
                                W[(k_block*2+(k-2*k_block))*n + i, r] = z[i, s]
                    r += 1
    
    # Actually, let's use the cleaner version that's known to work
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    r = 0
    for ab in range(2):
        for bg in range(4):
            for s in range(7):
                for i in range(n):
                    for jp in range(2):
                        U[i*m + ab*2 + jp, r] = x[s, i*2 + jp]
                for jp in range(2):
                    for kp in range(2):
                        V[(ab*2+jp)*p + 2*bg + kp, r] = y[s, jp*2 + kp]
                for i in range(n):
                    for kp in range(2):
                        W[(2*bg+kp)*n + i, r] = z[i*2+kp, s]
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
