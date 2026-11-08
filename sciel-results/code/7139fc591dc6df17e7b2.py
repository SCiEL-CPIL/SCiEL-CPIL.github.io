import numpy as np

def make_strassen_222():
    # Returns U, V, W for 2x2x2 Strassen decomposition
    # U: 4x7, V: 4x7, W: 4x7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen terms
    # 1: (A00+A11)(B00+B11) -> C00, C11
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # 2: (A10+A11)B00 -> C10, C11
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1 # C10 +1, C11 -1
    
    # 3: A00(B01-B11) -> C01, C11
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1 # C01 +1, C11 +1
    
    # 4: A11(B10-B00) -> C10, C00
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[2,3]=1; W[0,3]=1 # C10 +1, C00 +1
    
    # 5: (A00+A01)B11 -> C01, C00
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=1; W[0,4]=-1 # C01 +1, C00 -1
    
    # 6: (A10-A00)(B00+B01) -> C11
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1 # C11 +1
    
    # 7: (A01-A11)(B10+B11) -> C00
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1 # C00 +1
    
    return U, V, W

def make_simple_nm1(n, m):
    # Returns U, V, W for nxmx1 simple decomposition
    # U: (n*m)x(n*m), V: (m*1)x(n*m), W: (1*n)x(n*m)
    rank = n * m
    U = np.zeros((n*m, rank))
    V = np.zeros((m, rank))
    W = np.zeros((n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
            
    return U, V, W

def construct():
    n, m, p = 2, 3, 9
    
    # 1. Build 2x2x2 (Strassen)
    U_222, V_222, W_222 = make_strassen_222()
    
    # 2. Build 2x2x1 (Simple)
    U_221, V_221, W_221 = make_simple_nm1(2, 2)
    
    # 3. Build 2x2x3 (Combine 222 and 221)
    # 222: k in {0,1}. 221: k in {2}.
    r_222 = U_222.shape[1]
    r_221 = U_221.shape[1]
    r_223 = r_222 + r_221
    
    U_223 = np.hstack([U_222, U_221])
    
    # V_223 size: 2*3=6 x r_223
    V_223 = np.zeros((6, r_223))
    for c in range(r_222):
        # V_222 index j*2+k -> V_223 index j*3+k
        for j in range(2):
            for k in range(2):
                V_223[j*3+k, c] = V_222[j*2+k, c]
                
    for c in range(r_221):
        # V_221 index j -> V_223 index j*3+2
        for j in range(2):
            V_223[j*3+2, r_222+c] = V_221[j, c]
            
    # W_223 size: 3*2=6 x r_223
    W_223 = np.zeros((6, r_223))
    for c in range(r_222):
        # W_222 index k*2+i -> W_223 index k*2+i
        W_223[0:4, c] = W_222[0:4, c]
        
    for c in range(r_221):
        # W_221 index i -> W_223 index 2*2+i = 4+i
        W_223[4:6, r_222+c] = W_221[0:2, c]
        
    # 4. Build 2x1x3 (Simple)
    U_213, V_213, W_213 = make_simple_nm1(2, 1)
    
    # 5. Build 2x3x3 (Combine 213 and 223)
    # 213: j=0. 223: j in {1,2}.
    r_213 = U_213.shape[1]
    r_223 = U_223.shape[1]
    r_233 = r_213 + r_223
    
    U_233 = np.zeros((6, r_233))
    # U_213 index i -> U_233 index i*3+0
    for i in range(2):
        U_233[i*3+0, 0:r_213] = U_213[i, 0:r_213]
        
    # U_223 index i*2+j_loc -> U_233 index i*3+(1+j_loc)
    for i in range(2):
        for j_loc in range(2):
            U_233[i*3+(1+j_loc), r_213:r_213+r_223] = U_223[i*2+j_loc, 0:r_223]
            
    V_233 = np.zeros((9, r_233))
    # V_213 index k -> V_233 index 0*3+k
    for k in range(3):
        V_233[k, 0:r_213] = V_213[k, 0:r_213]
        
    # V_223 index j_loc*3+k -> V_233 index (1+j_loc)*3+k
    for j_loc in range(2):
        for k in range(3):
            V_233[(1+j_loc)*3+k, r_213:r_213+r_223] = V_223[j_loc*3+k, 0:r_223]
            
    W_233 = np.zeros((6, r_233))
    # W_213 index k*2+i -> W_233 index k*2+i
    W_233[:, 0:r_213] = W_213[:, 0:r_213]
    
    # W_223 index k*2+i -> W_233 index k*2+i
    W_233[:, r_213:r_213+r_223] = W_223[:, 0:r_223]
    
    # 6. Build 2x3x9 (Combine 3 blocks of 233)
    r_233 = U_233.shape[1]
    r_total = r_233 * 3
    
    U_final = np.zeros((6, r_total))
    V_final = np.zeros((27, r_total))
    W_final = np.zeros((18, r_total))
    
    for b in range(3):
        k_off = b * 3
        col_start = b * r_233
        col_end = col_start + r_233
        
        # U is same
        U_final[:, col_start:col_end] = U_233
        
        # V mapping
        for j in range(3):
            for k_loc in range(3):
                row_v_233 = j * 3 + k_loc
                row_v_final = j * 9 + (k_off + k_loc)
                V_final[row_v_final, col_start:col_end] = V_233[row_v_233, 0:r_233]
                
        # W mapping
        for k_loc in range(3):
            for i in range(2):
                row_w_233 = k_loc * 2 + i
                row_w_final = (k_off + k_loc) * 2 + i
                W_final[row_w_final, col_start:col_end] = W_233[row_w_233, 0:r_233]
                
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
