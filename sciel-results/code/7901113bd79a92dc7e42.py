import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # N=2 Strassen decomposition
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 3: M3 = A00(B10-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 4: M4 = A11(B01-B00)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    block_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0]),
        ([1, 0, 0, 0], [0, 0, 1, -1]),
        ([0, 0, 0, 1], [0, 1, -1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 0, 1, -1], [1, 1, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1])
    ]
    
    W_block = np.zeros((7, 4))
    W_block[0, 0] = 1; W_block[0, 3] = 1
    W_block[1, 2] = 1; W_block[1, 3] = -1
    W_block[2, 1] = 1; W_block[2, 3] = 1
    W_block[3, 0] = 1; W_block[3, 2] = 1
    W_block[4, 0] = -1; W_block[4, 1] = 1
    W_block[5, 3] = 1
    W_block[6, 0] = 1
    
    col = 0
    for t in range(7):
        U_coeffs, V_coeffs = block_terms[t]
        for s in range(7):
            for i in range(2):
                for j in range(2):
                    a_ij = U2[i*2+j, s]
                    if a_ij == 0: continue
                    for I in range(2):
                        for J in range(2):
                            u_coeff = U_coeffs[I*2+J]
                            if u_coeff == 0: continue
                            u_idx = (I*2+i)*4 + (J*2+j)
                            U4_val = a_ij * u_coeff
                            for j2 in range(2):
                                for k in range(2):
                                    b_jk = V2[j2*2+k, s]
                                    if b_jk == 0: continue
                                    for J2 in range(2):
                                        for K in range(2):
                                            v_coeff = V_coeffs[J2*2+K]
                                            if v_coeff == 0: continue
                                            if J != J2: continue
                                            v_idx = (J2*2+j2)*4 + (K*2+k)
                                            V4_val = b_jk * v_coeff
                                            w_idx = (K*2+k)*4 + (I*2+i)
                                            w_block_coeff = W_block[t, I*2+K]
                                            if w_block_coeff == 0: continue
                                            w2_coeff = W2[k*2+i, s]
                                            if w2_coeff == 0: continue
                                            W4_val = w_block_coeff * w2_coeff
                                            U[u_idx, col] += U4_val
                                            V[v_idx, col] += V4_val
                                            W[w_idx, col] += W4_val
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
