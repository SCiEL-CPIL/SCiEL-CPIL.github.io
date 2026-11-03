import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Rank 14 decomposition using two Strassen blocks
    rank = 14
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 * 2x8
    # U_sub (4, 7), V_sub (16, 7), W_sub (16, 7)
    U_sub = np.zeros((4, 7))
    V_sub = np.zeros((16, 7))
    W_sub = np.zeros((16, 7))
    
    # M0: (A00+A11)(B0+B1)
    U_sub[0, 0] = 1; U_sub[3, 0] = 1
    for k in range(8):
        V_sub[k, 0] = 1
        V_sub[8+k, 0] = 1
        W_sub[k*2, 0] = 1
        W_sub[k*2+1, 0] = 1
        
    # M1: (A10+A11)B0
    U_sub[2, 1] = 1; U_sub[3, 1] = 1
    for k in range(8):
        V_sub[k, 1] = 1
        W_sub[k*2+1, 1] = 1
        
    # M2: A00(B0-B1)
    U_sub[0, 2] = 1
    for k in range(8):
        V_sub[k, 2] = 1
        V_sub[8+k, 2] = -1
        W_sub[k*2, 2] = 1
        
    # M3: A11(B1-B0)
    U_sub[3, 3] = 1
    for k in range(8):
        V_sub[8+k, 3] = 1
        V_sub[k, 3] = -1
        W_sub[k*2, 3] = -1
        
    # M4: (A00+A01)B1
    U_sub[0, 4] = 1; U_sub[1, 4] = 1
    for k in range(8):
        V_sub[8+k, 4] = 1
        W_sub[k*2, 4] = 1
        
    # M5: (A10-A00)(B0+B1)
    U_sub[2, 5] = 1; U_sub[0, 5] = -1
    for k in range(8):
        V_sub[k, 5] = 1
        V_sub[8+k, 5] = 1
        W_sub[k*2, 5] = 1
        
    # M6: (A01-A11)(B1+B0)
    U_sub[1, 6] = 1; U_sub[3, 6] = -1
    for k in range(8):
        V_sub[8+k, 6] = 1
        V_sub[k, 6] = 1
        W_sub[k*2+1, 6] = 1
        
    # Assemble for 2x4 * 4x8
    # Group 0: j in {0,1}
    for c in range(7):
        # U: i in 0,1; j in 0,1 -> index i*4 + j
        for i in range(2):
            for j in range(2):
                U[i*m + j, c] = U_sub[i*2+j, c]
        # V: j in 0,1; k in 0..7 -> index j*8 + k
        for j in range(2):
            for k in range(8):
                V[j*p + k, c] = V_sub[j*8+k, c]
        # W: k in 0..7; i in 0,1 -> index k*2 + i
        for k in range(8):
            for i in range(2):
                W[k*n + i, c] = W_sub[k*2+i, c]
                
    # Group 1: j in {2,3}
    for c in range(7):
        # U: i in 0,1; j in 2,3 -> index i*4 + (j+2)
        for i in range(2):
            for j in range(2):
                U[i*m + (j+2), 7+c] = U_sub[i*2+j, c]
        # V: j in 2,3; k in 0..7 -> index (j+2)*8 + k
        for j in range(2):
            for k in range(8):
                V[(j+2)*p + k, 7+c] = V_sub[j*8+k, c]
        # W: k in 0..7; i in 0,1 -> index k*2 + i
        for k in range(8):
            for i in range(2):
                W[k*n + i, 7+c] = W_sub[k*2+i, c]
                
    # Verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid and time.time() < deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback
    rank_fb = n * m * p
    U_fb = [[0.0] * rank_fb for _ in range(n * m)]
    V_fb = [[0.0] * rank_fb for _ in range(m * p)]
    W_fb = [[0.0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1.0
                V_fb[j * p + k][col] = 1.0
                W_fb[k * n + i][col] = 1.0
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
