import numpy as np

def construct():
    n, m, p = 4, 4, 5
    rank_total = 53
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # 2x2 Strassen factors
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    X_combos = []
    Y_combos = []
    
    X0 = np.zeros((4,4)); X0[0:2, 0:2] = 1; X0[2:4, 2:4] = 1
    Y0 = np.zeros((4,4)); Y0[0:2, 0:2] = 1; Y0[2:4, 2:4] = 1
    X_combos.append(X0); Y_combos.append(Y0)
    
    X1 = np.zeros((4,4)); X1[2:4, 0:2] = 1; X1[2:4, 2:4] = 1
    Y1 = np.zeros((4,4)); Y1[0:2, 0:2] = 1
    X_combos.append(X1); Y_combos.append(Y1)
    
    X2 = np.zeros((4,4)); X2[0:2, 0:2] = 1
    Y2 = np.zeros((4,4)); Y2[0:2, 2:4] = 1; Y2[2:4, 2:4] = -1
    X_combos.append(X2); Y_combos.append(Y2)
    
    X3 = np.zeros((4,4)); X3[2:4, 2:4] = 1
    Y3 = np.zeros((4,4)); Y3[2:4, 0:2] = 1; Y3[0:2, 0:2] = -1
    X_combos.append(X3); Y_combos.append(Y3)
    
    X4 = np.zeros((4,4)); X4[0:2, 0:2] = 1; X4[0:2, 2:4] = 1
    Y4 = np.zeros((4,4)); Y4[2:4, 2:4] = 1
    X_combos.append(X4); Y_combos.append(Y4)
    
    X5 = np.zeros((4,4)); X5[2:4, 0:2] = 1; X5[0:2, 0:2] = -1
    Y5 = np.zeros((4,4)); Y5[0:2, 0:2] = 1; Y5[0:2, 2:4] = 1
    X_combos.append(X5); Y_combos.append(Y5)
    
    X6 = np.zeros((4,4)); X6[0:2, 2:4] = 1; X6[2:4, 2:4] = -1
    Y6 = np.zeros((4,4)); Y6[2:4, 0:2] = 1; Y6[2:4, 2:4] = 1
    X_combos.append(X6); Y_combos.append(Y6)
    
    # Outer Strassen W coefficients for C00, C01, C10, C11
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    W_outer = np.zeros((4, 7)) # 4 blocks, 7 terms
    W_outer[0, 0] = 1; W_outer[0, 3] = 1; W_outer[0, 4] = -1; W_outer[0, 6] = 1
    W_outer[1, 2] = 1; W_outer[1, 4] = 1
    W_outer[2, 1] = 1; W_outer[2, 3] = 1
    W_outer[3, 0] = 1; W_outer[3, 1] = -1; W_outer[3, 2] = 1; W_outer[3, 5] = 1

    col = 0
    for s in range(7):
        Xs = X_combos[s]
        Ys = Y_combos[s]
        
        X00 = Xs[0:2, 0:2]; X01 = Xs[0:2, 2:4]
        X10 = Xs[2:4, 0:2]; X11 = Xs[2:4, 2:4]
        
        Y00 = Ys[0:2, 0:2]; Y01 = Ys[0:2, 2:4]
        Y10 = Ys[2:4, 0:2]; Y11 = Ys[2:4, 2:4]
        
        inner_U = [
            X00 + X11,
            X10 + X11,
            X00,
            X11,
            X00 + X01,
            X10 - X00,
            X01 - X11
        ]
        inner_V = [
            Y00 + Y11,
            Y00,
            Y01 - Y11,
            Y10 - Y00,
            Y11,
            Y00 + Y01,
            Y10 + Y11
        ]
        
        for t in range(7):
            Ust = inner_U[t]
            Vst = inner_V[t]
            
            for r in range(7):
                R = col
                u_u = np.zeros((4, 4))
                v_v = np.zeros((4, 4))
                
                for i_prime in range(2):
                    for j_prime in range(2):
                        u_coeff = U2[i_prime*2 + j_prime, r]
                        if u_coeff != 0:
                            u_u[i_prime, j_prime] += X00[i_prime, j_prime] * u_coeff
                            u_u[i_prime, j_prime+2] += X01[i_prime, j_prime] * u_coeff
                            u_u[i_prime+2, j_prime] += X10[i_prime, j_prime] * u_coeff
                            u_u[i_prime+2, j_prime+2] += X11[i_prime, j_prime] * u_coeff
                            
                for j_prime in range(2):
                    for k_prime in range(2):
                        v_coeff = V2[j_prime*2 + k_prime, r]
                        if v_coeff != 0:
                            v_v[j_prime, k_prime] += Y00[j_prime, k_prime] * v_coeff
                            v_v[j_prime, k_prime+2] += Y01[j_prime, k_prime] * v_coeff
                            v_v[j_prime+2, k_prime] += Y10[j_prime, k_prime] * v_coeff
                            v_v[j_prime+2, k_prime+2] += Y11[j_prime, k_prime] * v_coeff
                            
                # W
                for k_prime in range(2):
                    for i_prime in range(2):
                        w_coeff = W2[k_prime*2 + i_prime, r]
                        if w_coeff != 0:
                            # Block 00
                            W[i_prime*4 + k_prime, R] += W_outer[0, s] * w_coeff
                            # Block 01
                            W[(i_prime)*4 + (k_prime+2), R] += W_outer[1, s] * w_coeff
                            # Block 10
                            W[(i_prime+2)*4 + k_prime, R] += W_outer[2, s] * w_coeff
                            # Block 11
                            W[(i_prime+2)*4 + (k_prime+2), R] += W_outer[3, s] * w_coeff
                            
                for i in range(4):
                    for j in range(4):
                        U[i*4+j, R] = u_u[i, j]
                        
                for j in range(4):
                    for k in range(4):
                        V[j*5+k, R] = v_v[j, k]
                        
                col += 1
                
    # Part 2: 4x4x1
    for i in range(4):
        for j in range(4):
            R = col
            U[i*4+j, R] = 1
            V[j*5+4, R] = 1
            W[4*4+i, R] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
