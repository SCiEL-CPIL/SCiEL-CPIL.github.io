import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 6, 6, 6
    
    # Strassen 2x2 (Rank 7)
    U_str = np.zeros((4, 7), dtype=int)
    V_str = np.zeros((4, 7), dtype=int)
    W_str = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11 + A22)(B11 + B22)
    U_str[0, 0] = 1; U_str[3, 0] = 1
    V_str[0, 0] = 1; V_str[3, 0] = 1
    W_str[0, 0] = 1; W_str[3, 0] = 1
    
    # M2 = (A21 + A22)B11
    U_str[2, 1] = 1; U_str[3, 1] = 1
    V_str[0, 1] = 1
    W_str[1, 1] = 1; W_str[3, 1] = -1
    
    # M3 = A11(B21 - B11)
    U_str[0, 2] = 1
    V_str[1, 2] = 1; V_str[3, 2] = -1
    W_str[2, 2] = 1; W_str[3, 2] = 1
    
    # M4 = A22(B12 - B22)
    U_str[3, 3] = 1
    V_str[0, 3] = -1; V_str[2, 3] = 1
    W_str[0, 3] = 1; W_str[1, 3] = 1
    
    # M5 = (A11 + A12)B22
    U_str[0, 4] = 1; U_str[1, 4] = 1
    V_str[3, 4] = 1
    W_str[0, 4] = -1; W_str[2, 4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U_str[0, 5] = -1; U_str[2, 5] = 1
    V_str[0, 5] = 1; V_str[1, 5] = 1
    W_str[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U_str[1, 6] = 1; U_str[3, 6] = -1
    V_str[2, 6] = 1; V_str[3, 6] = 1
    W_str[0, 6] = 1

    # Laderman 3x3 (Rank 23)
    # We need to define U_lad, V_lad, W_lad for 3x3 matrix multiplication.
    # Shapes: (9, 23)
    U_lad = np.zeros((9, 23), dtype=int)
    V_lad = np.zeros((9, 23), dtype=int)
    W_lad = np.zeros((9, 23), dtype=int)
    
    # Indices for 3x3: 0..8
    # Row i, Col j -> idx = i*3 + j
    
    # m1 = x11 y11
    U_lad[0, 0] = 1
    V_lad[0, 0] = 1
    W_lad[0, 0] = 1 # C11
    
    # m2 = (x12 + x13) y13
    U_lad[1, 1] = 1; U_lad[2, 1] = 1
    V_lad[2, 1] = 1
    W_lad[2, 1] = 1 # C13
    
    # m3 = x13 (y12 - y13)
    U_lad[2, 2] = 1
    V_lad[1, 2] = 1; V_lad[2, 2] = -1
    W_lad[1, 2] = 1 # C12
    
    # m4 = x21 y11
    U_lad[3, 3] = 1
    V_lad[0, 3] = 1
    W_lad[3, 3] = 1 # C21
    
    # m5 = x11 (y21 - y11)
    U_lad[0, 4] = 1
    V_lad[3, 4] = 1; V_lad[0, 4] = -1
    W_lad[3, 4] = 1 # C21
    
    # m6 = (x21 + x22) y22
    U_lad[3, 5] = 1; U_lad[4, 5] = 1
    V_lad[4, 5] = 1
    W_lad[4, 5] = 1 # C22
    
    # m7 = (x22 - x21) (y12 + y22)
    U_lad[3, 6] = -1; U_lad[4, 6] = 1
    V_lad[1, 6] = 1; V_lad[4, 6] = 1
    W_lad[4, 6] = 1 # C22
    
    # m8 = (x22 + x23) (y21 - y22)
    U_lad[4, 7] = 1; U_lad[5, 7] = 1
    V_lad[3, 7] = 1; V_lad[4, 7] = -1
    W_lad[5, 7] = 1 # C23
    
    # m9 = x23 (y22 + y23)
    U_lad[5, 8] = 1
    V_lad[4, 8] = 1; V_lad[5, 8] = 1
    W_lad[5, 8] = 1 # C23
    
    # m10 = (x31 - x21) y11
    U_lad[3, 9] = -1; U_lad[6, 9] = 1
    V_lad[0, 9] = 1
    W_lad[6, 9] = 1 # C31
    
    # m11 = (x31 + x32) y12
    U_lad[6, 10] = 1; U_lad[7, 10] = 1
    V_lad[1, 10] = 1
    W_lad[6, 10] = 1 # C31
    
    # m12 = (x32 - x31) (y11 + y12)
    U_lad[6, 11] = -1; U_lad[7, 11] = 1
    V_lad[0, 11] = 1; V_lad[1, 11] = 1
    W_lad[6, 11] = 1 # C31
    
    # m13 = x33 (y12 - y11)
    U_lad[8, 12] = 1
    V_lad[1, 12] = 1; V_lad[0, 12] = -1
    W_lad[7, 12] = 1 # C32
    
    # m14 = (x31 + x33) y13
    U_lad[6, 13] = 1; U_lad[8, 13] = 1
    V_lad[2, 13] = 1
    W_lad[7, 13] = 1 # C32
    
    # m15 = (x33 - x31) (y13 + y23)
    U_lad[6, 14] = -1; U_lad[8, 14] = 1
    V_lad[2, 14] = 1; V_lad[5, 14] = 1
    W_lad[7, 14] = 1 # C32
    
    # m16 = x33 (y21 - y23)
    U_lad[8, 15] = 1
    V_lad[3, 15] = 1; V_lad[5, 15] = -1
    W_lad[8, 15] = 1 # C33
    
    # m17 = (x31 + x32) y22
    U_lad[6, 16] = 1; U_lad[7, 16] = 1
    V_lad[4, 16] = 1
    W_lad[8, 16] = 1 # C33
    
    # m18 = (x32 - x33) (y22 + y23)
    U_lad[7, 17] = 1; U_lad[8, 17] = -1
    V_lad[4, 17] = 1; V_lad[5, 17] = 1
    W_lad[8, 17] = 1 # C33
    
    # m19 = x33 (y31 + y33)
    U_lad[8, 18] = 1
    V_lad[6, 18] = 1; V_lad[8, 18] = 1
    W_lad[8, 18] = 1 # C33
    
    # m20 = (x31 + x32) (y31 - y33)
    U_lad[6, 19] = 1; U_lad[7, 19] = 1
    V_lad[6, 19] = 1; V_lad[8, 19] = -1
    W_lad[8, 19] = 1 # C33
    
    # m21 = (x32 + x33) y33
    U_lad[7, 20] = 1; U_lad[8, 20] = 1
    V_lad[8, 20] = 1
    W_lad[8, 20] = 1 # C33
    
    # m22 = (x33 - x32) (y31 + y32)
    U_lad[7, 21] = -1; U_lad[8, 21] = 1
    V_lad[6, 21] = 1; V_lad[7, 21] = 1
    W_lad[8, 21] = 1 # C33
    
    # m23 = x33 (y31 - y32)
    U_lad[8, 22] = 1
    V_lad[6, 22] = 1; V_lad[7, 22] = -1
    W_lad[8, 22] = 1 # C33

    # Combine Strassen (2x2 blocks) and Laderman (3x3 blocks)
    # Total Rank = 7 * 23 = 161
    rank = 161
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    for r_s in range(7):
        for r_l in range(23):
            c = r_s * 23 + r_l
            if time.time() > deadline:
                break
                
            # U construction
            # U_str[I*2+J, r_s] is coeff for block A[I,J]
            # U_lad[i*3+j, r_l] is coeff for element X[i,j]
            # U[R*6+C, c] = U_str[I*2+J, r_s] * U_lad[i*3+j, r_l]
            # R = I*3 + i, C = J*3 + j
            for I in range(2):
                for J in range(2):
                    u_str_val = U_str[I*2+J, r_s]
                    if u_str_val != 0:
                        for i in range(3):
                            for j in range(3):
                                u_lad_val = U_lad[i*3+j, r_l]
                                if u_lad_val != 0:
                                    R = I*3 + i
                                    C = J*3 + j
                                    U[R*6+C, c] = u_str_val * u_lad_val
                                    
            # V construction
            # V_str[J*2+K, r_s] is coeff for block B[J,K]
            # V_lad[j*3+k, r_l] is coeff for element Y[j,k]
            # V[C*6+K_glob, c] = V_str[J*2+K, r_s] * V_lad[j*3+k, r_l]
            # C = J*3 + j, K_glob = K*3 + k
            for J in range(2):
                for K in range(2):
                    v_str_val = V_str[J*2+K, r_s]
                    if v_str_val != 0:
                        for j in range(3):
                            for k in range(3):
                                v_lad_val = V_lad[j*3+k, r_l]
                                if v_lad_val != 0:
                                    C = J*3 + j
                                    K_glob = K*3 + k
                                    V[C*6+K_glob, c] = v_str_val * v_lad_val
                                    
            # W construction
            # W_str[K*2+I, r_s] is coeff for block C[K,I]
            # W_lad[k*3+i, r_l] is coeff for element Z[k,i]
            # W[K_glob*6+R, c] = W_str[K*2+I, r_s] * W_lad[k*3+i, r_l]
            # K_glob = K*3 + k, R = I*3 + i
            for K in range(2):
                for I in range(2):
                    w_str_val = W_str[K*2+I, r_s]
                    if w_str_val != 0:
                        for k in range(3):
                            for i in range(3):
                                w_lad_val = W_lad[k*3+i, r_l]
                                if w_lad_val != 0:
                                    K_glob = K*3 + k
                                    R = I*3 + i
                                    W[K_glob*6+R, c] = w_str_val * w_lad_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
