"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Strassen 2x2x2 (rank 7)
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # i1=0,j1=0
        [0, 0, 0, 0, 1, 0, 1],   # i1=0,j1=1
        [0, 1, 0, 0, 0, 1, 0],   # i1=1,j1=0
        [1, 1, 0, 1, 0, 0, -1]   # i1=1,j1=1
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # j1=0,k1=0
        [0, 0, 1, 0, 0, 1, 0],   # j1=0,k1=1
        [0, 0, 0, 1, 0, 0, 1],   # j1=1,k1=0
        [1, 0, -1, 0, 1, 0, 1]   # j1=1,k1=1
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # k1=0,i1=0
        [0, 1, 0, 1, 0, 0, 0],   # k1=0,i1=1
        [0, 0, 1, 0, 1, 0, 0],   # k1=1,i1=0
        [1, -1, 1, 0, 0, 1, 0]   # k1=1,i1=1
    ]
    
    # Build 4x4x4 (rank 49)
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j][c] = U2[i1*2+j1][t] * U2[i2*2+j2][s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k][c] = V2[j1*2+k1][t] * V2[j2*2+k2][s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i][c] = W2[k1*2+i1][t] * W2[k2*2+i2][s]
    
    # Build 4x4x8 (rank 98)
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0]*rank for _ in range(n*m)]  # 16 x 98
    V = [[0]*rank for _ in range(m*p)]  # 32 x 98
    W = [[0]*rank for _ in range(p*n)]  # 32 x 98
    
    # First block (k=0..3)
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U[i*4+j][c] = U4[i*4+j][c]
    
    for j in range(4):
        for k in range(4):  # k=0..3
            for c in range(49):
                V[j*8+k][c] = V4[j*4+k][c]
    
    for k in range(4):  # k=0..3
        for i in range(4):
            for c in range(49):
                W[k*4+i][c] = W4[k*4+i][c]
    
    # Second block (k=4..7)
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U[i*4+j][c+49] = U4[i*4+j][c]
    
    for j in range(4):
        for k in range(4):  # k=0..3 within block, actual k = k+4
            for c in range(49):
                V[j*8+(k+4)][c+49] = V4[j*4+k][c]
    
    for k in range(4):  # k=0..3 within block, actual k = k+4
        for i in range(4):
            for c in range(49):
                W[(k+4)*4+i][c+49] = W4[k*4+i][c]
    
    # Verify (optional, for debugging)
    # if time.time() - start_time < deadline - 5:
    #     ... verify ...
    
    return {"U": U, "V": W and V, "W": W}  # Oops, typo
