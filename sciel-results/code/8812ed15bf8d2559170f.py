import time
import sys

def construct():
    n, m, p = 3, 3, 8
    # Base rank is 72. We replace one 2x2x2 block with Strassen (rank 7).
    # The block is i in {0,1}, j in {0,1}, k in {0,1}.
    # We use 7 Strassen columns and 64 schoolbook columns.
    rank = 7 + 64
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part (indices 0 to 6)
    # U, V, W indices for Strassen
    # i in {0,1}, j in {0,1} -> U[i*3+j, r]
    # j in {0,1}, k in {0,1} -> V[j*8+k, r]
    # k in {0,1}, i in {0,1} -> W[k*3+i, r]
    
    # r=0: m1 = (A00+A11)(B00+B11)
    U[0*3+0][0] = 1
    U[1*3+1][0] = 1
    V[0*8+0][0] = 1
    V[1*8+1][0] = 1
    W[0*3+0][0] = 1 # C00
    W[1*3+1][0] = 1 # C11
    
    # r=1: m2 = (A10+A11)B00
    U[1*3+0][1] = 1
    U[1*3+1][1] = 1
    V[0*8+0][1] = 1
    W[0*3+1][1] = 1 # C10
    W[1*3+1][1] = -1 # C11
    
    # r=2: m3 = A00(B10-B11)
    U[0*3+0][2] = 1
    V[1*8+0][2] = 1
    V[1*8+1][2] = -1
    W[1*3+0][2] = 1 # C01
    
    # r=3: m4 = A11B11
    U[1*3+1][3] = 1
    V[1*8+1][3] = 1
    W[0*3+1][3] = 1 # C10
    W[1*3+1][3] = 1 # C11
    
    # r=4: m5 = (A00+A01)(B11-B00)
    U[0*3+0][4] = 1
    U[0*3+1][4] = 1
    V[1*8+1][4] = 1
    V[0*8+0][4] = -1
    W[1*3+0][4] = 1 # C01
    W[0*3+0][4] = -1 # C00
    
    # r=5: m6 = (A10-A00)(B00+B01)
    U[1*3+0][5] = 1
    U[0*3+0][5] = -1
    V[0*8+0][5] = 1
    V[0*8+1][5] = 1
    W[1*3+1][5] = 1 # C11
    
    # r=6: m7 = (A01-A11)(B10+B11)
    U[0*3+1][6] = 1
    U[1*3+1][6] = -1
    V[1*8+0][6] = 1
    V[1*8+1][6] = 1
    W[1*3+0][6] = 1 # C01
    W[0*3+0][6] = 1 # C00

    # Schoolbook part (indices 7 to 70)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the Strassen block
                if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
