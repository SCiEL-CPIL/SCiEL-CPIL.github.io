import numpy as np
import time

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    rank = 56  # 8 independent 2x2x2 blocks, each rank 7 (Strassen)
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 56
    V = [[0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * rank for _ in range(p * n)]   # 32 x 56
    
    # Strassen rank-7 decomposition of 2x2 matrix multiplication
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    u_vecs = [
        [1, 0, 0, 1],    # M1 = (A00+A11)(B00-B11)
        [0, 0, 1, 1],    # M2 = (A10+A11)B00
        [1, 0, 0, 0],    # M3 = A00(B01-B11)
        [0, 0, 0, 1],    # M4 = A11 B11
        [1, 1, 0, 0],    # M5 = (A00+A01)B11
        [-1, 0, 1, 0],   # M6 = (A10-A00)(B00+B01)
        [0, 1, 0, -1]    # M7 = (A01-A11)(B10+B11)
    ]
    v_vecs = [
        [1, 0, 0, -1],   # M1: B00 - B11
        [1, 0, 0, 0],    # M2: B00
        [0, 1, 0, -1],   # M3: B01 - B11
        [0, 0, 0, 1],    # M4: B11
        [0, 0, 0, 1],    # M5: B11
        [1, 1, 0, 0],    # M6: B00 + B01
        [0, 0, 1, 1]     # M7: B10 + B11
    ]
    # W vectors: output C = M1 - M3 - M6 + M7 (C00), M5 (C01), M2 + M4 (C10), M4 (C11)
    # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    w_vecs = [
        [1, 0, 0, 0],    # M1 contributes to C00
        [0, 0, 1, 0],    # M2 contributes to C10
        [-1, 0, 0, 0],   # M3 contributes -1 to C00
        [0, 0, 1, 1],    # M4 contributes to C10 and C11
        [0, 1, 0, 0],    # M5 contributes to C01
        [-1, 0, 0, 0],   # M6 contributes -1 to C00
        [1, 0, 0, 0]     # M7 contributes to C00
    ]
    
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            
            # Fill U: A indices 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U[li][col] = u_vecs[s][li]
            
            # Fill V: B block r indices
            # Local 0->B[0,2r]=V[2r], 1->B[0,2r+1]=V[2r+1]
            # Local 2->B[1,2r]=V[16+2r], 3->B[1,2r+1]=V[16+2r+1]
            v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V[v_idx[li]][col] = v_vecs[s][li]
            
            # Fill W: C block r indices (trace convention W[k*n+i])
            # Local 0->C[0,2r]=W[(2r)*2+0]=W[4r]
            # Local 1->C[0,2r+1]=W[(2r+1)*2+0]=W[4r+2]
            # Local 2->C[1,2r]=W[(2r)*2+1]=W[4r+1]
            # Local 3->C[1,2r+1]=W[(2r+1)*2+1]=W[4r+3]
            w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W[w_idx[li]][col] = w_vecs[s][li]
    
    return {"U": U, "V": V, "W": W}
