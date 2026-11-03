import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2x2
    # U_in: 4x7, V_in: 4x7, W_in: 4x7
    # Indices: i,j in {0,1} -> idx i*2+j
    #           j,k in {0,1} -> idx j*2+k
    #           k,i in {0,1} -> idx k*2+i
    
    U_in = np.zeros((4, 7))
    V_in = np.zeros((4, 7))
    W_in = np.zeros((4, 7))
    
    # M0: (x00+x11)(y00+y11)
    U_in[0, 0] = 1; U_in[3, 0] = 1
    V_in[0, 0] = 1; V_in[3, 0] = 1
    W_in[0, 0] = 1; W_in[3, 0] = 1 # z00, z11
    
    # M1: (x10+x11)y00
    U_in[2, 1] = 1; U_in[3, 1] = 1
    V_in[0, 1] = 1
    W_in[2, 1] = 1; W_in[3, 1] = -1 # z10, z11
    
    # M2: x00(y01-y11)
    U_in[0, 2] = 1
    V_in[1, 2] = 1; V_in[3, 2] = -1
    W_in[1, 2] = 1; W_in[3, 2] = 1 # z01, z11
    
    # M3: x11(y10-y00)
    U_in[3, 3] = 1
    V_in[2, 3] = 1; V_in[0, 3] = -1
    W_in[0, 3] = 1; W_in[2, 3] = 1 # z00, z10
    
    # M4: (x00+x01)y11
    U_in[0, 4] = 1; U_in[1, 4] = 1
    V_in[3, 4] = 1
    W_in[0, 4] = -1; W_in[1, 4] = 1 # z00, z01
    
    # M5: (x10-x00)(y00+y01)
    U_in[2, 5] = 1; U_in[0, 5] = -1
    V_in[0, 5] = 1; V_in[1, 5] = 1
    W_in[3, 5] = 1 # z11
    
    # M6: (x01-x11)(y10+y11)
    U_in[1, 6] = 1; U_in[3, 6] = -1
    V_in[2, 6] = 1; V_in[3, 6] = 1
    W_in[0, 6] = 1 # z00

    # Rank 49 decomposition for 4x4x4
    # U_out: 4x7 (blocks 00, 01, 10, 11), V_out: 4x7, W_out: 4x7
    U_out = np.zeros((4, 7))
    V_out = np.zeros((4, 7))
    W_out = np.zeros((4, 7))
    
    # M0: (A00+A11)(B00+B11)
    U_out[0, 0] = 1; U_out[3, 0] = 1
    V_out[0, 0] = 1; V_out[3, 0] = 1
    W_out[0, 0] = 1; W_out[3, 0] = 1
    
    # M1: (A10+A11)B00
    U_out[2, 1] = 1; U_out[3, 1] = 1
    V_out[0, 1] = 1
    W_out[2, 1] = 1; W_out[3, 1] = -1
    
    # M2: A00(B01-B11)
    U_out[0, 2] = 1
    V_out[1, 2] = 1; V_out[3, 2] = -1
    W_out[1, 2] = 1; W_out[3, 2] = 1
    
    # M3: A11(B10-B00)
    U_out[3, 3] = 1
    V_out[2, 3] = 1; V_out[0, 3] = -1
    W_out[0, 3] = 1; W_out[2, 3] = 1
    
    # M4: (A00+A01)B11
    U_out[0, 4] = 1; U_out[1, 4] = 1
    V_out[3, 4] = 1
    W_out[0, 4] = -1; W_out[1, 4] = 1
    
    # M5: (A10-A00)(B00+B01)
    U_out[2, 5] = 1; U_out[0, 5] = -1
    V_out[0, 5] = 1; V_out[1, 5] = 1
    W_out[3, 5] = 1
    
    # M6: (A01-A11)(B10+B11)
    U_out[1, 6] = 1; U_out[3, 6] = -1
    V_out[2, 6] = 1; V_out[3, 6] = 1
    W_out[0, 6] = 1

    # Construct 4x4x4 factors (rank 49)
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    for c_out in range(7):
        for c_in in range(7):
            c = c_out * 7 + c_in
            for I in range(2):
                for J in range(2):
                    u_out_val = U_out[I*2+J, c_out]
                    if u_out_val == 0: continue
                    for i_prime in range(2):
                        for j_prime in range(2):
                            i = I*2 + i_prime
                            j = J*2 + j_prime
                            u_in_val = U_in[i_prime*2+j_prime, c_in]
                            U_44[i*4+j, c] = u_out_val * u_in_val
                            
    for c_out in range(7):
        for c_in in range(7):
            c = c_out * 7 + c_in
            for J in range(2):
                for K in range(2):
                    v_out_val = V_out[J*2+K, c_out]
                    if v_out_val == 0: continue
                    for j_prime in range(2):
                        for k_prime in range(2):
                            j = J*2 + j_prime
                            k = K*2 + k_prime
                            v_in_val = V_in[j_prime*2+k_prime, c_in]
                            V_44[j*4+k, c] = v_out_val * v_in_val
                            
    for c_out in range(7):
        for c_in in range(7):
            c = c_out * 7 + c_in
            for K in range(2):
                for I in range(2):
                    w_out_val = W_out[K*2+I, c_out]
                    if w_out_val == 0: continue
                    for k_prime in range(2):
                        for i_prime in range(2):
                            k = K*2 + k_prime
                            i = I*2 + i_prime
                            w_in_val = W_in[k_prime*2+i_prime, c_in]
                            W_44[k*4+i, c] = w_out_val * w_in_val

    # Add rank 16 for the 5th column (k=4)
    # Total rank 65
    U_45 = np.zeros((16, 65))
    V_45 = np.zeros((20, 65))
    W_45 = np.zeros((20, 65))
    
    # Copy 4x4x4 part
    U_45[:, :49] = U_44
    V_45[:, :49] = V_44
    W_45[:, :49] = W_44
    
    # 5th column: k=4
    # C_i4 = sum_j A_ij B_j4
    for i in range(4):
        for j in range(4):
            c = 49 + i*4 + j
            U_45[i*4+j, c] = 1.0
            V_45[j*5+4, c] = 1.0
            W_45[4*4+i, c] = 1.0
            
    return {
        "U": U_45.tolist(),
        "V": V_45.tolist(),
        "W": W_45.tolist()
    }

