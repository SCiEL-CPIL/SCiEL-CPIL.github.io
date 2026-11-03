import numpy as np
import time

def construct():
    deadline = time.time() + 145
    
    # M_2,2,2 Strassen Decomposition (Rank 7)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # Term 0: (A00+A11)(B00+B11)
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # Term 1: (A10+A11)B00
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1
    
    # Term 2: A00(B10-B11)
    U_222[0, 2] = 1
    V_222[2, 2] = 1; V_222[3, 2] = -1
    W_222[2, 2] = 1
    
    # Term 3: A11(B01-B00)
    U_222[3, 3] = 1
    V_222[1, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1; W_222[2, 3] = 1
    
    # Term 4: (A00+A01)B11
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = -1; W_222[2, 4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[3, 5] = 1
    
    # Term 6: (A01-A11)(B10+B11)
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1; W_222[3, 6] = -1

    # M_2,2,1 Schoolbook Decomposition (Rank 4)
    # U: (4, 4), V: (2, 4), W: (2, 4)
    # Indices: U[i*2+j, c], V[j*1+k, c], W[k*2+i, c] (k=0)
    U_221 = np.zeros((4, 4))
    V_221 = np.zeros((2, 4))
    W_221 = np.zeros((2, 4))
    
    # c=0: A00 B00 -> C00
    U_221[0, 0] = 1; V_221[0, 0] = 1; W_221[0, 0] = 1
    # c=1: A01 B10 -> C00
    U_221[1, 1] = 1; V_221[1, 1] = 1; W_221[0, 1] = 1
    # c=2: A10 B00 -> C10
    U_221[2, 2] = 1; V_221[0, 2] = 1; W_221[1, 2] = 1
    # c=3: A11 B10 -> C10
    U_221[3, 3] = 1; V_221[1, 3] = 1; W_221[1, 3] = 1

    def kron(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R = R1 * R2
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        for c1 in range(R1):
            for c2 in range(R2):
                c = c1 * R2 + c2
                # U
                for i1 in range(n1):
                    for j1 in range(m1):
                        u1_val = U1[i1*m1+j1, c1]
                        if u1_val == 0: continue
                        for i2 in range(n2):
                            for j2 in range(m2):
                                u2_val = U2[i2*m2+j2, c2]
                                if u2_val == 0: continue
                                u_val = u1_val * u2_val
                                i = i1 * n2 + i2
                                j = j1 * m2 + j2
                                U[i*m+j, c] += u_val
                # V
                for j1 in range(m1):
                    for k1 in range(p1):
                        v1_val = V1[j1*p1+k1, c1]
                        if v1_val == 0: continue
                        for j2 in range(m2):
                            for k2 in range(p2):
                                v2_val = V2[j2*p2+k2, c2]
                                if v2_val == 0: continue
                                v_val = v1_val * v2_val
                                j = j1 * m2 + j2
                                k = k1 * p2 + k2
                                V[j*p+k, c] += v_val
                # W
                for k1 in range(p1):
                    for i1 in range(n1):
                        w1_val = W1[k1*n1+i1, c1]
                        if w1_val == 0: continue
                        for k2 in range(p2):
                            for i2 in range(n2):
                                w2_val = W2[k2*n2+i2, c2]
                                if w2_val == 0: continue
                                w_val = w1_val * w2_val
                                k = k1 * p2 + k2
                                i = i1 * n2 + i2
                                W[k*n+i, c] += w_val
                                
        return U, V, W

    def direct_sum(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R = R1 + R2
        p = p1 + p2
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Copy Part 1
        U[:, :R1] = U1
        for j in range(m):
            for k in range(p1):
                V[j*p+k, :R1] = V1[j*p1+k, :R1]
        for k in range(p1):
            for i in range(n):
                W[k*n+i, :R1] = W1[k*n1+i, :R1] # Note: n1 is n here
                
        # Copy Part 2
        U[:, R1:] = U2
        for j in range(m):
            for k in range(p2):
                V[j*p + p1 + k, R1:] = V2[j*p2+k, R1:]
        for k in range(p2):
            for i in range(n):
                W[(p1+k)*n+i, R1:] = W2[k*n+i, R1:]
                
        return U, V, W

    # M_4,4,4 = M_2,2,2 (Kron) M_2,2,2
    U_444, V_444, W_444 = kron(U_222, V_222, W_222, U_222, V_222, W_222, 2, 2, 2, 2, 2, 2)
    
    # M_2,2,3 = M_2,2,2 (Direct Sum) M_2,2,1
    U_223, V_223, W_223 = direct_sum(U_222, V_222, W_222, U_221, V_221, W_221, 2, 2, 2, 1)
    
    # M_4,4,3 = M_2,2,3 (Kron) M_2,2,1
    U_443, V_443, W_443 = kron(U_223, V_223, W_223, U_221, V_221, W_221, 2, 2, 3, 2, 2, 1)
    
    # M_4,4,7 = M_4,4,4 (Direct Sum) M_4,4,3
    U_final, V_final, W_final = direct_sum(U_444, V_444, W_444, U_443, V_443, W_443, 4, 4, 4, 3)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
