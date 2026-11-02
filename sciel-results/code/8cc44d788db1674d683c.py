import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct Strassen 2x2x2 decomposition (Rank 7)
    # U, V, W are 4x7 matrices
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11) -> C00+C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10+C11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3: A00(B01-B11) -> C01+C11
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00+C10
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5: (A00+A01)B11 -> -C00+C01
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # 2. Construct Strassen 4x4x4 decomposition (Rank 49)
    # U4, V4, W4 are 16x49 matrices
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r_out in range(7):
        for r_in in range(7):
            r = r_out * 7 + r_in
            # U4[i_global, j_global, r] = sum_{I,J} U_out[I,J,r_out] * U_in[i,j,r_in]
            # i_global = I*2 + i, j_global = J*2 + j
            for I in range(2):
                for J in range(2):
                    coeff_u = U2[I*2+J, r_out] # Wait, U2 indices are i*2+j?
                    # My U2 definition: U2[i*2+j, r]?
                    # In step 1, I used U2[0,0] for A00. A00 is i=0,j=0. Index 0*2+0=0.
                    # A11 is i=1,j=1. Index 1*2+1=3.
                    # So U2 index is i*2+j.
                    if coeff_u != 0:
                        for i in range(2):
                            for j in range(2):
                                coeff_in = U2[i*2+j, r_in]
                                if coeff_in != 0:
                                    i_global = I*2 + i
                                    j_global = J*2 + j
                                    U4[i_global*4 + j_global, r] += coeff_u * coeff_in
                                
            # V4[j_global, k_global, r] = sum_{J,K} V_out[J,K,r_out] * V_in[j,k,r_in]
            for J in range(2):
                for K in range(2):
                    coeff_v = V2[J*2+K, r_out]
                    if coeff_v != 0:
                        for j in range(2):
                            for k in range(2):
                                coeff_in = V2[j*2+k, r_in]
                                if coeff_in != 0:
                                    j_global = J*2 + j
                                    k_global = K*2 + k
                                    V4[j_global*4 + k_global, r] += coeff_v * coeff_in
                                    
            # W4[k_global, i_global, r] = sum_{K,I} W_out[K,I,r_out] * W_in[k,i,r_in]
            for K in range(2):
                for I in range(2):
                    coeff_w = W2[K*2+I, r_out]
                    if coeff_w != 0:
                        for k in range(2):
                            for i in range(2):
                                coeff_in = W2[k*2+i, r_in]
                                if coeff_in != 0:
                                    k_global = K*2 + k
                                    i_global = I*2 + i
                                    W4[k_global*4 + i_global, r] += coeff_w * coeff_in

    # 3. Construct Schoolbook 4x4x1 decomposition (Rank 16)
    U1 = np.zeros((16, 16))
    V1 = np.zeros((4, 16)) # m*p = 4*1 = 4
    W1 = np.zeros((4, 16)) # p*n = 1*4 = 4
    
    col = 0
    for i in range(4):
        for j in range(4):
            U1[i*4+j, col] = 1
            V1[j, col] = 1
            W1[i, col] = 1 # k=0, so k*4+i = i
            col += 1
            
    # 4. Combine for 4x4x9
    total_rank = 49 + 49 + 16
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((36, total_rank)) # m*p = 4*9 = 36
    W_total = np.zeros((36, total_rank)) # p*n = 9*4 = 36
    
    # Block 1: Cols 0-3
    U_total[:, 0:49] = U4
    for r in range(49):
        for j in range(4):
            for k in range(4):
                val = V4[j*4+k, r]
                if val != 0:
                    V_total[j*9+k, r] = val
        for k in range(4):
            for i in range(4):
                val = W4[k*4+i, r]
                if val != 0:
                    W_total[k*4+i, r] = val
    # Note: W_total index is k_global*4 + i_global.
    # In block 1, k_global = k (0-3).
    
    # Block 2: Cols 4-7
    U_total[:, 49:98] = U4
    for r in range(49):
        for j in range(4):
            for k in range(4):
                val = V4[j*4+k, r]
                if val != 0:
                    V_total[j*9+(k+4), 49+r] = val
        for k in range(4):
            for i in range(4):
                val = W4[k*4+i, r]
                if val != 0:
                    W_total[(k+4)*4+i, 49+r] = val

    # Block 3: Col 8
    U_total[:, 98:114] = U1
    for r in range(16):
        for j in range(4):
            val = V1[j, r]
            if val != 0:
                V_total[j*9+8, 98+r] = val
        for i in range(4):
            val = W1[i, r]
            if val != 0:
                W_total[8*4+i, 98+r] = val

    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
