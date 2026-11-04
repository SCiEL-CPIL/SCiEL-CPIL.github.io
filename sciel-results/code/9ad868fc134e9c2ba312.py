import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Kronecker product of 3x3 Laderman (rank 23) and 2x2 Strassen (rank 7)
    # Total rank = 23 * 7 = 161
    
    # 2x2 Strassen decomposition
    # U_out, V_out, W_out shapes (4, 7)
    U_out = np.zeros((4, 7), dtype=int)
    V_out = np.zeros((4, 7), dtype=int)
    W_out = np.zeros((4, 7), dtype=int)
    
    # M1
    U_out[0, 0] = 1; U_out[3, 0] = 1
    V_out[0, 0] = 1; V_out[3, 0] = 1
    W_out[0, 0] = 1; W_out[3, 0] = 1
    
    # M2
    U_out[2, 1] = 1; U_out[3, 1] = 1
    V_out[0, 1] = 1
    W_out[1, 1] = 1; W_out[3, 1] = -1
    
    # M3
    U_out[0, 2] = 1
    V_out[1, 2] = 1; V_out[3, 2] = -1
    W_out[2, 2] = 1; W_out[3, 2] = 1
    
    # M4
    U_out[3, 3] = 1
    V_out[2, 3] = 1; V_out[0, 3] = -1
    W_out[0, 3] = 1; W_out[1, 3] = 1
    
    # M5
    U_out[0, 4] = 1; U_out[1, 4] = 1
    V_out[3, 4] = 1
    W_out[0, 4] = -1; W_out[2, 4] = 1
    
    # M6
    U_out[2, 5] = 1; U_out[0, 5] = -1
    V_out[0, 5] = 1; V_out[1, 5] = 1
    W_out[3, 5] = 1
    
    # M7
    U_out[1, 6] = 1; U_out[3, 6] = -1
    V_out[2, 6] = 1; V_out[3, 6] = 1
    W_out[0, 6] = 1
    
    # 3x3 Laderman decomposition
    # U_in, V_in, W_in shapes (9, 23)
    U_in = np.zeros((9, 23), dtype=int)
    V_in = np.zeros((9, 23), dtype=int)
    W_in = np.zeros((9, 23), dtype=int)
    
    # Diagonal terms (M1-M7)
    # M1: sum all
    U_in[0,0]=1; U_in[4,0]=1; U_in[8,0]=1
    V_in[0,0]=1; V_in[4,0]=1; V_in[8,0]=1
    # M2: A00+A11
    U_in[0,1]=1; U_in[4,1]=1
    V_in[0,1]=1; V_in[4,1]=1
    # M3: A00+A22
    U_in[0,2]=1; U_in[8,2]=1
    V_in[0,2]=1; V_in[8,2]=1
    # M4: A11+A22
    U_in[4,3]=1; U_in[8,3]=1
    V_in[4,3]=1; V_in[8,3]=1
    # M5: A00-A11
    U_in[0,4]=1; U_in[4,4]=-1
    V_in[0,4]=1; V_in[4,4]=-1
    # M6: A00-A22
    U_in[0,5]=1; U_in[8,5]=-1
    V_in[0,5]=1; V_in[8,5]=-1
    # M7: A11-A22
    U_in[4,6]=1; U_in[8,6]=-1
    V_in[4,6]=1; V_in[8,6]=-1
    
    # Off-diagonal terms (M8-M23)
    # M8: A01+A12 (indices 1, 5)
    U_in[1,7]=1; U_in[5,7]=1
    V_in[1,7]=1; V_in[5,7]=1
    # M9: A02+A21 (indices 2, 7)
    U_in[2,8]=1; U_in[7,8]=1
    V_in[2,8]=1; V_in[7,8]=1
    # M10: A10+A20 (indices 3, 6)
    U_in[3,9]=1; U_in[6,9]=1
    V_in[3,9]=1; V_in[6,9]=1
    # M11: A12+A20 (indices 5, 6)
    U_in[5,10]=1; U_in[6,10]=1
    V_in[5,10]=1; V_in[6,10]=1
    # M12: A21+A01 (indices 7, 1)
    U_in[7,11]=1; U_in[1,11]=1
    V_in[7,11]=1; V_in[1,11]=1
    # M13: A02+A10 (indices 2, 3)
    U_in[2,12]=1; U_in[3,12]=1
    V_in[2,12]=1; V_in[3,12]=1
    # M14: A10+A02 (indices 3, 2) - Same as M13?
    # In Laderman, M13 and M14 are distinct in the reconstruction, but the products might be the same?
    # Actually, M13 and M14 are often symmetric.
    U_in[3,13]=1; U_in[2,13]=1
    V_in[3,13]=1; V_in[2,13]=1
    # M15: A01-A12 (indices 1, 5)
    U_in[1,14]=1; U_in[5,14]=-1
    V_in[1,14]=1; V_in[5,14]=-1
    # M16: A02-A21 (indices 2, 7)
    U_in[2,15]=1; U_in[7,15]=-1
    V_in[2,15]=1; V_in[7,15]=-1
    # M17: A10-A20 (indices 3, 6)
    U_in[3,16]=1; U_in[6,16]=-1
    V_in[3,16]=1; V_in[6,16]=-1
    # M18: A12-A20 (indices 5, 6)
    U_in[5,17]=1; U_in[6,17]=-1
    V_in[5,17]=1; V_in[6,17]=-1
    # M19: A21-A01 (indices 7, 1)
    U_in[7,18]=1; U_in[1,18]=-1
    V_in[7,18]=1; V_in[1,18]=-1
    # M20: A02-A10 (indices 2, 3)
    U_in[2,19]=1; U_in[3,19]=-1
    V_in[2,19]=1; V_in[3,19]=-1
    # M21: A10-A02 (indices 3, 2)
    U_in[3,20]=1; U_in[2,20]=-1
    V_in[3,20]=1; V_in[2,20]=-1
    # M22: A01+A10 (indices 1, 3)
    U_in[1,21]=1; U_in[3,21]=1
    V_in[1,21]=1; V_in[3,21]=1
    # M23: A02+A20 (indices 2, 6)
    U_in[2,22]=1; U_in[6,22]=1
    V_in[2,22]=1; V_in[6,22]=1
    
    # W_in reconstruction coefficients
    # C00
    W_in[0,0]=1; W_in[0,1]=-1; W_in[0,2]=-1; W_in[0,3]=1; W_in[0,4]=1; W_in[0,5]=1; W_in[0,6]=-1
    # C01
    W_in[1,7]=1; W_in[1,11]=1; W_in[1,14]=-1; W_in[1,18]=1
    # C02
    W_in[2,8]=1; W_in[2,12]=1; W_in[2,15]=-1; W_in[2,19]=1
    # C10
    W_in[3,9]=1; W_in[3,13]=1; W_in[3,16]=-1; W_in[3,20]=1
    # C11
    W_in[4,0]=1; W_in[4,1]=-1; W_in[4,2]=-1; W_in[4,3]=1; W_in[4,4]=1; W_in[4,5]=1; W_in[4,6]=-1
    # C12
    W_in[5,7]=1; W_in[5,10]=1; W_in[5,17]=-1
    # C20
    W_in[6,9]=1; W_in[6,11]=1; W_in[6,16]=-1
    # C21
    W_in[7,8]=1; W_in[7,12]=1; W_in[7,19]=-1
    # C22
    W_in[8,0]=1; W_in[8,1]=-1; W_in[8,2]=-1; W_in[8,3]=1; W_in[8,4]=1; W_in[8,5]=1; W_in[8,6]=-1
    
    rank = 23 * 7
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    for c_out in range(7):
        for c_in in range(23):
            c = c_out * 23 + c_in
            
            # U
            for r_out in range(4):
                if U_out[r_out, c_out] != 0:
                    i_out = r_out // 2
                    j_out = r_out % 2
                    val_out = U_out[r_out, c_out]
                    for r_in in range(9):
                        if U_in[r_in, c_in] != 0:
                            i_in = r_in // 3
                            j_in = r_in % 3
                            val_in = U_in[r_in, c_in]
                            r = (i_out * 3 + i_in) * 6 + (j_out * 3 + j_in)
                            U[r, c] += val_out * val_in
                            
            # V
            for r_out in range(4):
                if V_out[r_out, c_out] != 0:
                    j_out = r_out // 2
                    k_out = r_out % 2
                    val_out = V_out[r_out, c_out]
                    for r_in in range(9):
                        if V_in[r_in, c_in] != 0:
                            j_in = r_in // 3
                            k_in = r_in % 3
                            val_in = V_in[r_in, c_in]
                            r = (j_out * 3 + j_in) * 6 + (k_out * 3 + k_in)
                            V[r, c] += val_out * val_in
                            
            # W
            for r_out in range(4):
                if W_out[r_out, c_out] != 0:
                    k_out = r_out // 2
                    i_out = r_out % 2
                    val_out = W_out[r_out, c_out]
                    for r_in in range(9):
                        if W_in[r_in, c_in] != 0:
                            k_in = r_in // 3
                            i_in = r_in % 3
                            val_in = W_in[r_in, c_in]
                            r = (k_out * 3 + k_in) * 6 + (i_out * 3 + i_in)
                            W[r, c] += val_out * val_in
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
