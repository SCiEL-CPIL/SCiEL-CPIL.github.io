"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook rank 64
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Try Rank 49 via Kronecker product of Strassen
    # Strassen 2x2x2 decomposition
    # U2[i*2+j][c], V2[j*2+k][c], W2[k*2+i][c]
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 0, 1, 0, 1, 0],   # B01
        [0, 0, 1, 0, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    W2 = [
        [1, 0, 0, -1, 0, 1, -1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, 0, -1, 0, 1, -1, 0]  # C11
    ]
    
    # Verify 2x2 Strassen decomposition
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # If Strassen is wrong, return schoolbook
        return schoolbook()

    # Construct Rank 49 decomposition for 4x4x4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # Iterate over all i, j, k in 4x4
            # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            # U[i*m+j] = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
                            U[i * 4 + j][c] = U2[i1 * 2 + j1][c1] * U2[i2 * 2 + j2][c2]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    # V[j*p+k] = V2[j1*2+k1][c1] * V2[j2*2+k2][c2]
                                    V[j * 4 + k][c] = V2[j1 * 2 + k1][c1] * V2[j2 * 2 + k2][c2]
                                    # W[k*n+i] = W2[k1*2+i1][c1] * W2[k2*2+i2][c2]
                                    W[k * 4 + i][c] = W2[k1 * 2 + i1][c1] * W2[k2 * 2 + i2][c2]
                                    
    # Verify 4x4 decomposition
    # sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 1
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U[i*4+j][c] * V[j*4+k][c] * W[k*4+i][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        return schoolbook()

    if time.time() > deadline:
        return schoolbook()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
