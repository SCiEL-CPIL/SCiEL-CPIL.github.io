"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """Attempt Strassen-based rank-49 decomposition; fall back to schoolbook if invalid."""
    import numpy as np
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # 2x2 Strassen decomposition factors
    # U2[i*2+j, r]: A factor
    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0],  # a
        [0,  0,  0,  0,  1,  0, -1],  # b
        [0,  1,  0,  0,  0,  1,  0],  # c
        [1,  1,  0,  1,  0,  0,  1],  # d
    ], dtype=float)
    
    # V2[j*2+k, r]: B factor
    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0],  # e
        [0,  0, -1,  0,  0,  1,  1],  # f
        [0,  0,  0,  1,  0,  1,  1],  # g
        [1,  0,  1,  0,  1,  0,  0],  # h
    ], dtype=float)
    
    # W2[k*2+i, r]: C factor (k=col, i=row)
    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1],  # C[0,0]
        [0,  1,  0,  1,  0,  0,  0],  # C[1,0] (k=0,i=1)
        [0,  0,  1,  0,  1,  0,  0],  # C[0,1] (k=1,i=0)
        [1, -1, 1,  0,  0,  1,  0],  # C[1,1]
    ], dtype=float)
    
    # Verify 2x2: sum_r U2[i*2+j,r] * V2[j*2+k,r] * W2[k*2+i,r] == 1 for all i,j,k
    valid_2x2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U2[i*2+j] * V2[j*2+k] * W2[k*2+i])
                if abs(val - 1.0) > 1e-10:
                    valid_2x2 = False
                    break
        if not valid_2x2:
            break
    
    if valid_2x2:
        # Build 4x4 via Kronecker product
        rank = 49
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for i in range(n):
            for j in range(m):
                i1, i2 = i // 2, i % 2
                j1, j2 = j // 2, j % 2
                for r1 in range(7):
                    for r2 in range(7):
                        r = r1 * 7 + r2
                        U[i*m + j, r] = U2[i1*2 + i2, r1] * U2[j1*2 + j2, r2]
        
        for j in range(m):
            for k in range(p):
                j1, j2 = j // 2, j % 2
                k1, k2 = k // 2, k % 2
                for r1 in range(7):
                    for r2 in range(7):
                        r = r1 * 7 + r2
                        V[j*p + k, r] = V2[j1*2 + j2, r1] * V2[k1*2 + k2, r2]
        
        for k in range(p):
            for i in range(n):
                k1, k2 = k // 2, k % 2
                i1, i2 = i // 2, i % 2
                for r1 in range(7):
                    for r2 in range(7):
                        r = r1 * 7 + r2
                        W[k*n + i, r] = W2[k1*2 + k2, r1] * W2[i1*2 + i2, r2]
        
        # Verify 4x4 decomposition
        if time.time() < deadline - 10:
            valid_4x4 = True
            for i in range(n):
                if time.time() > deadline - 10:
                    valid_4x4 = False
                    break
                for j in range(m):
                    if time.time() > deadline - 10:
                        valid_4x4 = False
                        break
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        if abs(val - 1.0) > 1e-10:
                            valid_4x4 = False
                            break
                    if not valid_4x4:
                        break
                if not valid_4x4:
                    break
            
            if valid_4x4:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 64
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
# EVOLVE-BLOCK-END
