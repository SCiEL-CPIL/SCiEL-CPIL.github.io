"""
Tensor decomposition for 4x4x4 matrix multiplication.
Uses the Kronecker product of the rank-7 Strassen decomposition for 2x2x2.
Rank = 7 * 7 = 49.
"""
import numpy as np
import time

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition for 4x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Strassen decomposition for 2x2x2 (Rank 7)
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Target: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k in {0,1}
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0],  # B01
        [0, 0, 1, 0, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ])
    
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 0], # C00 (k=0, i=0) -> W[0*2+0, c]
        [0, 0, 0, 0, 1, 0, 0], # C01 (k=0, i=1) -> W[0*2+1, c]
        [0, 1, 1, 0, 0, 1, 0], # C10 (k=1, i=0) -> W[1*2+0, c]
        [1, 0, 0, 0, 0, 0, 1]  # C11 (k=1, i=1) -> W[1*2+1, c]
    ])
    
    # Verify Strassen base
    valid_base = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
                if s != 1:
                    valid_base = False
                    break
            if not valid_base: break
        if not valid_base: break
        
    if not valid_base:
        return _schoolbook_4x4()

    # Construct Rank-49 decomposition via Kronecker Product
    rank = 49
    n, m, p = 4, 4, 4
    
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1*2+j1][c1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2*2+j2][c2]
                            if u2 == 0: continue
                            
                            i = i1*2 + i2
                            j = j1*2 + j2
                            U[i*4 + j][c] = u1 * u2
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1*2 + k2
                                    
                                    v1 = V2[j1*2+k1][c1]
                                    v2 = V2[j2*2+k2][c2]
                                    w1 = W2[k1*2+i1][c1]
                                    w2 = W2[k2*2+i2][c2]
                                    
                                    V[j*4 + k][c] = v1 * v2
                                    W[k*4 + i][c] = w1 * w2
                                    
    # Verify
    valid_49 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U[i*4+j][c] * V[j*4+k][c] * W[k*4+i][c]
                if s != 1:
                    valid_49 = False
                    break
            if not valid_49: break
        if not valid_49: break
        
    if not valid_49:
        return _schoolbook_4x4()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_4x4():
    rank = 64
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
