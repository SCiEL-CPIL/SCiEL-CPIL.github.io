import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local 2x2 Strassen factors
    # U_local[uv, t], V_local[vw, t], W_local[wu, t]
    # t in 0..6
    U_local = np.zeros((4, 7))
    V_local = np.zeros((4, 7))
    W_local = np.zeros((4, 7))
    
    # m1: (x00+x11)(y00+y11)
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    # W1 contributes to z00, z01, z10, z11
    # z00: +1, z01: 0, z10: 0, z11: +1? 
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    # W_local index is wu. z_wu.
    # w=0, u=0 -> z00. w=0, u=1 -> z01. w=1, u=0 -> z10. w=1, u=1 -> z11.
    W_local[0, 0] = 1  # z00
    W_local[3, 0] = 1  # z11
    
    # m2: (x10+x11)y00
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1  # z10
    W_local[3, 1] = -1 # z11
    
    # m3: x00(y01-y11)
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1  # z01
    W_local[3, 2] = 1  # z11
    
    # m4: x11(y10-y00)
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1  # z00
    W_local[2, 3] = 1  # z10
    
    # m5: (x00+x01)y11
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1 # z00
    W_local[1, 4] = 1  # z01
    
    # m6: (x10-x00)(y00+y01)
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1  # z11
    
    # m7: (x01-x11)(y10+y11)
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1  # z00
    
    # Block level coefficients
    # X_s[i,j] for A, Y_s[j,k] for B, Z_s[i,k] for C (coeff for M_s)
    # s in 0..6
    X_s = []
    Y_s = []
    Z_s = []
    
    # M1 = (A11+A22)(B11+B22)
    X1 = np.zeros((4,4)); X1[0,0]=1; X1[0,1]=1; X1[1,0]=1; X1[1,1]=1; X1[2,2]=1; X1[2,3]=1; X1[3,2]=1; X1[3,3]=1
    Y1 = np.zeros((4,4)); Y1[0,0]=1; Y1[0,1]=1; Y1[1,0]=1; Y1[1,1]=1; Y1[2,2]=1; Y1[2,3]=1; Y1[3,2]=1; Y1[3,3]=1
    Z1 = np.ones((4,4))
    X_s.append(X1); Y_s.append(Y1); Z_s.append(Z1)
    
    # M2 = (A21+A22)B11
    X2 = np.zeros((4,4)); X2[2,0]=1; X2[2,1]=1; X2[3,0]=1; X2[3,1]=1; X2[2,2]=1; X2[2,3]=1; X2[3,2]=1; X2[3,3]=1
    Y2 = np.zeros((4,4)); Y2[0,0]=1; Y2[0,1]=1; Y2[1,0]=1; Y2[1,1]=1
    Z2 = np.zeros((4,4)); Z2[2,0]=1; Z2[2,1]=1; Z2[3,0]=1; Z2[3,1]=1
    X_s.append(X2); Y_s.append(Y2); Z_s.append(Z2)
    
    # M3 = A11(B12-B22)
    X3 = np.zeros((4,4)); X3[0,0]=1; X3[0,1]=1; X3[1,0]=1; X3[1,1]=1
    Y3 = np.zeros((4,4)); Y3[0,2]=1; Y3[0,3]=1; Y3[1,2]=1; Y3[1,3]=1; Y3[2,2]=-1; Y3[2,3]=-1; Y3[3,2]=-1; Y3[3,3]=-1
    Z3 = np.zeros((4,4)); Z3[0,2]=1; Z3[0,3]=1; Z3[1,2]=1; Z3[1,3]=1
    X_s.append(X3); Y_s.append(Y3); Z_s.append(Z3)
    
    # M4 = A22(B21-B11)
    X4 = np.zeros((4,4)); X4[2,2]=1; X4[2,3]=1; X4[3,2]=1; X4[3,3]=1
    Y4 = np.zeros((4,4)); Y4[2,0]=1; Y4[2,1]=1; Y4[3,0]=1; Y4[3,1]=1; Y4[0,0]=-1; Y4[0,1]=-1; Y4[1,0]=-1; Y4[1,1]=-1
    Z4 = np.zeros((4,4)); Z4[0,0]=1; Z4[0,1]=1; Z4[1,0]=1; Z4[1,1]=1; Z4[2,0]=1; Z4[2,1]=1; Z4[3,0]=1; Z4[3,1]=1
    X_s.append(X4); Y_s.append(Y4); Z_s.append(Z4)
    
    # M5 = (A11+A12)B22
    X5 = np.zeros((4,4)); X5[0,0]=1; X5[0,1]=1; X5[1,0]=1; X5[1,1]=1; X5[0,2]=1; X5[0,3]=1; X5[1,2]=1; X5[1,3]=1
    Y5 = np.zeros((4,4)); Y5[2,2]=1; Y5[2,3]=1; Y5[3,2]=1; Y5[3,3]=1
    Z5 = np.zeros((4,4)); Z5[0,0]=-1; Z5[0,1]=-1; Z5[1,0]=-1; Z5[1,1]=-1; Z5[0,2]=1; Z5[0,3]=1; Z5[1,2]=1; Z5[1,3]=1
    X_s.append(X5); Y_s.append(Y5); Z_s.append(Z5)
    
    # M6 = (A21-A11)(B11+B12)
    X6 = np.zeros((4,4)); X6[2,0]=1; X6[2,1]=1; X6[3,0]=1; X6[3,1]=1; X6[0,0]=-1; X6[0,1]=-1; X6[1,0]=-1; X6[1,1]=-1
    Y6 = np.zeros((4,4)); Y6[0,0]=1; Y6[0,1]=1; Y6[1,0]=1; Y6[1,1]=1; Y6[0,2]=1; Y6[0,3]=1; Y6[1,2]=1; Y6[1,3]=1
    Z6 = np.zeros((4,4)); Z6[2,2]=1; Z6[2,3]=1; Z6[3,2]=1; Z6[3,3]=1
    X_s.append(X6); Y_s.append(Y6); Z_s.append(Z6)
    
    # M7 = (A12-A22)(B21+B22)
    X7 = np.zeros((4,4)); X7[0,2]=1; X7[0,3]=1; X7[1,2]=1; X7[1,3]=1; X7[2,2]=-1; X7[2,3]=-1; X7[3,2]=-1; X7[3,3]=-1
    Y7 = np.zeros((4,4)); Y7[2,0]=1; Y7[2,1]=1; Y7[3,0]=1; Y7[3,1]=1; Y7[2,2]=1; Y7[2,3]=1; Y7[3,2]=1; Y7[3,3]=1
    Z7 = np.zeros((4,4)); Z7[0,0]=1; Z7[0,1]=1; Z7[1,0]=1; Z7[1,1]=1
    X_s.append(X7); Y_s.append(Y7); Z_s.append(Z7)
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for i in range(4):
                for j in range(4):
                    u = i % 2
                    v = j % 2
                    U[i * 4 + j][r] = U_local[u * 2 + v][t] * X_s[s][i][j]
            for j in range(4):
                for k in range(4):
                    v = j % 2
                    w = k % 2
                    V[j * 4 + k][r] = V_local[v * 2 + w][t] * Y_s[s][j][k]
            for k in range(4):
                for i in range(4):
                    w = k % 2
                    u = i % 2
                    W[k * 4 + i][r] = W_local[w * 2 + u][t] * Z_s[s][i][k]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
