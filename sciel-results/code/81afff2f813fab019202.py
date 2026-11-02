"""Matrix multiplication 2x2 by 2x16 with Strassen-based rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    rank = 56  # 8 blocks * 7 Strassen steps
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen rank-7 decomposition vectors for 2x2 * 2x2
    # Local indices: A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    #                B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    #                C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    u_vecs = [
        [1, 0, 0, 1],   # M1: (A00 - A10) * (B00 + B10)
        [0, 0, 1, 1],   # M2: (A00 + A10) * B10
        [1, 0, 0, 0],   # M3: A01 * (B10 - B00)
        [0, 0, 0, 1],   # M4: A11 * (B00 - B10)
        [1, 1, 0, 0],   # M5: (A00 + A01) * B00
        [-1, 0, 1, 0],  # M6: (A10 - A11) * (B00 + B11)
        [0, 1, 0, -1]   # M7: (A01 + A11) * (B10 - B11)
    ]
    v_vecs = [
        [1, 0, 0, 1],   # M1: B00 + B10
        [1, 0, 0, 0],   # M2: B10
        [0, 1, 0, -1],  # M3: B10 - B00
        [-1, 0, 1, 0],  # M4: B00 - B10
        [0, 0, 0, 1],   # M5: B00
        [1, 1, 0, 0],   # M6: B00 + B11
        [0, 0, 1, 1]    # M7: B10 - B11
    ]
    w_vecs = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, -1],  # M2: C10 - C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ]
    
    # Split B into 8 blocks of 2 columns each
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            
            # U: A indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U[li][col] = u_vecs[s][li]
                    
            # V: B indices for block r
            # li=0 -> B[0,2r]=V[2r], li=1 -> B[0,2r+1]=V[2r+1]
            # li=2 -> B[1,2r]=V[16+2r], li=3 -> B[1,2r+1]=V[16+2r+1]
            v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V[v_idx[li]][col] = v_vecs[s][li]
                    
            # W: C indices for block r (trace convention W[k*n+i, r])
            # li=0 -> C[0,2r]=W[4r], li=1 -> C[0,2r+1]=W[4r+2]
            # li=2 -> C[1,2r]=W[4r+1], li=3 -> C[1,2r+1]=W[4r+3]
            w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W[w_idx[li]][col] = w_vecs[s][li]
                    
    return {"U": U, "V": V, "W": W}
