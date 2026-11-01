import numpy as np
import time

DIMENSIONS = [2, 2, 16]
_DEADLINE = 140.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen vectors
    u_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_s = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_s = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for b in range(num_blocks):
        col_offset = b * 7
        for s in range(7):
            c = col_offset + s
            
            # U: A indices
            U[0, c] += u_s[s][0]
            U[1, c] += u_s[s][1]
            U[2, c] += u_s[s][2]
            U[3, c] += u_s[s][3]
            
            # V: B indices for block b
            # B is 2x16. Block b is cols 2b, 2b+1.
            # V index: row * 16 + col
            # B[0, 2b] -> 0*16 + 2b = 2b
            # B[0, 2b+1] -> 2b+1
            # B[1, 2b] -> 16 + 2b
            # B[1, 2b+1] -> 16 + 2b + 1
            V[2*b, c] += v_s[s][0]
            V[2*b+1, c] += v_s[s][1]
            V[16+2*b, c] += v_s[s][2]
            V[16+2*b+1, c] += v_s[s][3]
            
            # W: C indices
            # C is 16x2? No, W[k*n+i, r].
            # Task says: W[k*n+i, r].
            # C is 2x16.
            # Standard trace convention?
            # "output factor follows the official trace convention W[k*n+i,r]"
            # Usually C[i,j] is stored.
            # If W is (p*n, rank), then index is j*n + i?
            # p=16, n=2.
            # C[0, 2b] -> index 2b*2 + 0 = 4b
            # C[0, 2b+1] -> index (2b+1)*2 + 0 = 4b+2
            # C[1, 2b] -> index 2b*2 + 1 = 4b+1
            # C[1, 2b+1] -> index (2b+1)*2 + 1 = 4b+3
            
            W[4*b, c] += w_s[s][0]
            W[4*b+2, c] += w_s[s][1]
            W[4*b+1, c] += w_s[s][2]
            W[4*b+3, c] += w_s[s][3]
            
    if time.time() - _t0 > _DEADLINE:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
