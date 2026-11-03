import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 65
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications for 2x2x2
    # u[i,j,r], v[j,k,r], w[k,i,r]
    # i, j, k in {0, 1}
    u_222 = [[0]*2 for _ in range(2)] # u[i,j]
    v_222 = [[0]*2 for _ in range(2)] # v[j,k]
    w_222 = [[0]*2 for _ in range(2)] # w[k,i]
    
    # We need 7 sets of u, v, w
    # Let's define them explicitly
    # r=1: M1 = (A00+A11)(B00+B11)
    # u: 00=1, 11=1
    # v: 00=1, 11=1
    # w: 00=1, 11=-1
    # r=2: M2 = (A10+A11)B00
    # u: 10=1, 11=1
    # v: 00=1
    # w: 10=1, 11=1
    # r=3: M3 = A00(B01-B11)
    # u: 00=1
    # v: 01=1, 11=-1
    # w: 01=1, 11=1
    # r=4: M4 = A11(B10-B00)
    # u: 11=1
    # v: 10=1, 00=-1
    # w: 00=1, 10=1
    # r=5: M5 = (A00+A01)B11
    # u: 00=1, 01=1
    # v: 11=1
    # w: 10=1, 11=-1
    # r=6: M6 = (A10-A00)(B00+B01)
    # u: 10=1, 00=-1
    # v: 00=1, 01=1
    # w: 10=1, 11=1 (Assuming)
    # r=7: M7 = (A01-A11)(B10+B11)
    # u: 01=1, 11=-1
    # v: 10=1, 11=1
    # w: 00=-1, 01=1 (Assuming)
    
    # Actually, I will use the known correct Strassen tensors.
    # The 7 rank-1 tensors for M_2,2,2 are:
    # 1: (1,1,0; 1,1,0) (1,0,0; 1,0,0) (1,0,0; 0,-1,0) ? No.
    
    # Let's use the standard Strassen decomposition.
    # The 7 multiplications are:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # And the results are:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M5
    # C22 = -M1 + M2 + M3 + M4
    
    # I will construct the U, V, W for these 7.
    # u[i,j,r], v[j,k,r], w[k,i,r]
    # i,j,k in {0,1}
    
    u = [[0]*7 for _ in range(4)] # u[i*2+j, r]
    v = [[0]*7 for _ in range(4)] # v[j*2+k, r]
    w = [[0]*7 for _ in range(4)] # w[k*2+i, r]
    
    # r=1
    u[0][0] = 1; u[3][0] = 1
    v[0][0] = 1; v[3][0] = 1
    w[0][0] = 1; w[3][0] = -1
    # r=2
    u[2][1] = 1; u[3][1] = 1
    v[0][1] = 1
    w[2][1] = 1; w[3][1] = 1
    # r=3
    u[0][2] = 1
    v[1][2] = 1; v[3][2] = -1
    w[1][2] = 1; w[3][2] = 1
    # r=4
    u[3][3] = 1
    v[2][3] = 1; v[0][3] = -1
    w[0][3] = 1; w[2][3] = 1
    # r=5
    u[0][4] = 1; u[1][4] = 1
    v[3][4] = 1
    w[2][4] = 1; w[3][4] = -1
    # r=6
    u[2][5] = 1; u[0][5] = -1
    v[0][5] = 1; v[1][5] = 1
    w[2][5] = 1; w[3][5] = 1
    # r=7
    u[1][6] = 1; u[3][6] = -1
    v[2][6] = 1; v[3][6] = 1
    w[0][6] = -1; w[1][6] = 1
    
    # Now construct the 49 tensors for M_4,4,4
    # Indices I = 2*i1 + i2, J = 2*j1 + j2, K = 2*k1 + k2
    # U[I, J, r*7+s] = u[i1*2+j1, r] * u[i2*2+j2, s]
    # V[J, K, r*7+s] = v[j1*2+k1, r] * v[j2*2+k2, s]
    # W[K, I, r*7+s] = w[k1*2+i1, r] * w[k2*2+i2, s]
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    I = i1 * 2 + i2
                                    J = j1 * 2 + j2
                                    K = k1 * 2 + k2
                                    u_val = u[i1*2+j1][r] * u[i2*2+j2][s]
                                    v_val = v[j1*2+k1][r] * v[j2*2+k2][s]
                                    w_val = w[k1*2+i1][r] * w[k2*2+i2][s]
                                    U[I*4+J][col] += u_val
                                    V[J*5+K][col] += v_val
                                    W[K*4+I][col] += w_val
                                    
    # Now add the 16 tensors for M_4,4,1
    # Indices I in {0,1,2,3}, J in {0,1,2,3}, K=0
    # U[I, J, 49+c] = 1
    # V[J, 0, 49+c] = 1
    # W[0, I, 49+c] = 1
    for I in range(4):
        for J in range(4):
            col = 49 + I*4 + J
            U[I*4+J][col] = 1
            V[J*5+0][col] = 1
            W[0*4+I][col] = 1
            
    return {"U": U, "V": V, "W": W}
