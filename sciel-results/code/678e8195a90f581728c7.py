import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 6 x 28
    V = [[0] * rank for _ in range(m * p)]   # 15 x 28
    W = [[0] * rank for _ in range(p * n)]   # 10 x 28
    
    r = 0
    
    # Part 1: A[:,2] * B[2,:] -> 2*5 = 10 terms
    # u_idx = i*m + 2, v_idx = 2*p + k, w_idx = k*n + i
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][r] = 1
            V[v_idx][r] = 1
            W[w_idx][r] = 1
            r += 1
    
    # Part 2: A[:,0:2] * B[0:2, 4] -> 2*2 = 4 terms
    # u_idx = i*m + j, v_idx = j*p + 4, w_idx = 4*n + i
    for i in range(n):
        for j in range(2):
            u_idx = i * m + j
            v_idx = j * p + 4
            w_idx = 4 * n + i
            U[u_idx][r] = 1
            V[v_idx][r] = 1
            W[w_idx][r] = 1
            r += 1
    
    # Part 3: A[:,0:2] * B[0:2, 0:2] -> 2x2x2 Strassen (7 terms)
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # u_idx = i*3 + j, v_idx = j*5 + k, w_idx = k*2 + i
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0*3+0][r] = 1; U[1*3+1][r] = 1
    V[0*5+0][r] = 1; V[1*5+1][r] = 1
    W[0*2+0][r] = 1; W[1*2+1][r] = 1
    r += 1
    
    # M2: (A10+A11)B00 -> C10, C11
    U[1*3+0][r] = 1; U[1*3+1][r] = 1
    V[0*5+0][r] = 1
    W[0*2+1][r] = 1; W[1*2+1][r] = -1
    r += 1
    
    # M3: A00(B11-B01) -> C10, C11
    U[0*3+0][r] = 1
    V[0*5+1][r] = 1; V[1*5+1][r] = -1
    W[1*2+0][r] = 1; W[1*2+1][r] = 1
    r += 1
    
    # M4: (A01+A11)(B00-B10) -> C00, C01
    U[1*3+1][r] = 1
    V[1*5+0][r] = 1; V[0*5+0][r] = -1
    W[0*2+0][r] = 1; W[0*2+1][r] = 1
    r += 1
    
    # M5: (A00+A01)B11 -> C00, C01
    U[0*3+0][r] = 1; U[0*3+1][r] = 1
    V[1*5+1][r] = 1
    W[0*2+0][r] = -1; W[1*2+0][r] = 1
    r += 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[1*3+0][r] = 1; U[0*3+0][r] = -1
    V[0*5+0][r] = 1; V[0*5+1][r] = 1
    W[1*2+1][r] = 1
    r += 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[0*3+1][r] = 1; U[1*3+1][r] = -1
    V[1*5+0][r] = 1; V[1*5+1][r] = 1
    W[0*2+0][r] = 1
    r += 1
    
    # Part 4: A[:,0:2] * B[0:2, 2:4] -> 2x2x2 Strassen (7 terms)
    # Indices: i in {0,1}, j in {0,1}, k in {2,3}
    # u_idx = i*3 + j, v_idx = j*5 + k, w_idx = k*2 + i
    
    # M1: (A00+A11)(B02+B13) -> C02, C13
    U[0*3+0][r] = 1; U[1*3+1][r] = 1
    V[0*5+2][r] = 1; V[1*5+3][r] = 1
    W[2*2+0][r] = 1; W[3*2+1][r] = 1
    r += 1
    
    # M2: (A10+A11)B02 -> C12, C13
    U[1*3+0][r] = 1; U[1*3+1][r] = 1
    V[0*5+2][r] = 1
    W[2*2+1][r] = 1; W[3*2+1][r] = -1
    r += 1
    
    # M3: A00(B13-B03) -> C12, C13
    U[0*3+0][r] = 1
    V[0*5+3][r] = 1; V[1*5+3][r] = -1
    W[3*2+0][r] = 1; W[3*2+1][r] = 1
    r += 1
    
    # M4: (A01+A11)(B02-B12) -> C02, C03
    U[1*3+1][r] = 1
    V[1*5+2][r] = 1; V[0*5+2][r] = -1
    W[2*2+0][r] = 1; W[2*2+1][r] = 1
    r += 1
    
    # M5: (A00+A01)B13 -> C02, C03
    U[0*3+0][r] = 1; U[0*3+1][r] = 1
    V[1*5+3][r] = 1
    W[2*2+0][r] = -1; W[3*2+0][r] = 1
    r += 1
    
    # M6: (A10-A00)(B02+B03) -> C13
    U[1*3+0][r] = 1; U[0*3+0][r] = -1
    V[0*5+2][r] = 1; V[0*5+3][r] = 1
    W[3*2+1][r] = 1
    r += 1
    
    # M7: (A01-A11)(B12+B13) -> C02
    U[0*3+1][r] = 1; U[1*3+1][r] = -1
    V[1*5+2][r] = 1; V[1*5+3][r] = 1
    W[2*2+0][r] = 1
    r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
