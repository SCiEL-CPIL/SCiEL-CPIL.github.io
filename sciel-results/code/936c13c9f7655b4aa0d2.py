import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # We'll use block Strassen: 4 blocks of 2x2x2, each with rank 7
    # Total rank = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    col = 0
    
    # Helper: add a Strassen block
    # X = A sub-block (2x2), Y = B sub-block (2x2), Z contributes to C sub-block
    # x_indices: list of 4 U-indices for X entries [x00, x01, x10, x11]
    # y_indices: list of 4 V-indices for Y entries [y00, y01, y10, y11]
    # z_indices: list of 4 W-indices for Z entries [z00, z01, z10, z11]
    # z_sign: +1 or -1 multiplier for Z contribution
    
    def add_strassen(x_idx, y_idx, z_idx, sign=1):
        nonlocal col
        # x_idx = [x00, x01, x10, x11] (U row indices)
        # y_idx = [y00, y01, y10, y11] (V row indices)
        # z_idx = [z00, z01, z10, z11] (W row indices)
        
        # M1 = (x00+x11)*(y00+y11)
        U[x_idx[0]][col] += 1
        U[x_idx[3]][col] += 1
        V[y_idx[0]][col] += 1
        V[y_idx[3]][col] += 1
        W[z_idx[0]][col] += sign  # z00
        W[z_idx[3]][col] += sign  # z11
        col += 1
        
        # M2 = (x10+x11)*y00
        U[x_idx[2]][col] += 1
        U[x_idx[3]][col] += 1
        V[y_idx[0]][col] += 1
        W[z_idx[2]][col] += sign  # z10
        col += 1
        
        # M3 = x00*(y10-y11)
        U[x_idx[0]][col] += 1
        V[y_idx[2]][col] += 1
        V[y_idx[3]][col] -= 1
        W[z_idx[1]][col] += sign  # z01
        W[z_idx[3]][col] += sign  # z11
        col += 1
        
        # M4 = x11*(y01-y00)
        U[x_idx[3]][col] += 1
        V[y_idx[1]][col] += 1
        V[y_idx[0]][col] -= 1
        W[z_idx[0]][col] += sign  # z00
        W[z_idx[2]][col] += sign  # z10
        col += 1
        
        # M5 = (x00+x01)*y11
        U[x_idx[0]][col] += 1
        U[x_idx[1]][col] += 1
        V[y_idx[3]][col] += 1
        W[z_idx[0]][col] -= sign  # z00
        W[z_idx[1]][col] += sign  # z01
        col += 1
        
        # M6 = (x10-x00)*(y00+y10)
        U[x_idx[2]][col] += 1
        U[x_idx[0]][col] -= 1
        V[y_idx[0]][col] += 1
        V[y_idx[2]][col] += 1
        W[z_idx[3]][col] += sign  # z11
        col += 1
        
        # M7 = (x01-x11)*(y01+y11)
        U[x_idx[1]][col] += 1
        U[x_idx[3]][col] -= 1
        V[y_idx[1]][col] += 1
        V[y_idx[3]][col] += 1
        W[z_idx[0]][col] += sign  # z00
        col += 1
    
    # A indices: A[i,j] -> U[i*4+j]
    # A_L = A[:, 0:2]: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[4], A[1,1]=U[5]
    # A_R = A[:, 2:4]: A[0,2]=U[2], A[0,3]=U[3], A[1,2]=U[6], A[1,3]=U[7]
    
    # B indices: B[j,k] -> V[j*4+k]
    # B_TL = B[0:2, 0:2]: B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[4], B[1,1]=V[5]
    # B_TR = B[0:2, 2:4]: B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[6], B[1,3]=V[7]
    # B_BL = B[2:4, 0:2]: B[2,0]=V[8], B[2,1]=V[9], B[3,0]=V[12], B[3,1]=V[13]
    # B_BR = B[2:4, 2:4]: B[2,2]=V[10], B[2,3]=V[11], B[3,2]=V[14], B[3,3]=V[15]
    
    # C indices: C[i,k] -> W[k*2+i]
    # C_L = C[:, 0:2]: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    # C_R = C[:, 2:4]: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    
    # Block 1: S1 = A_L * B_TL -> C_L
    # X = A_L: [x00,x01,x10,x11] = [U[0], U[1], U[4], U[5]]
    # Y = B_TL: [y00,y01,y10,y11] = [V[0], V[1], V[4], V[5]]
    # Z = C_L: [z00,z01,z10,z11] = [W[0], W[2], W[1], W[3]]
    add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3])
    
    # Block 2: S2 = A_R * B_BL -> C_L
    # X = A_R: [x00,x01,x10,x11] = [U[2], U[3], U[6], U[7]]
    # Y = B_BL: [y00,y01,y10,y11] = [V[8], V[9], V[12], V[13]]
    # Z = C_L: [z00,z01,z10,z11] = [W[0], W[2], W[1], W[3]]
    add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3])
    
    # Block 3: S3 = A_L * B_TR -> C_R
    # X = A_L: [x00,x01,x10,x11] = [U[0], U[1], U[4], U[5]]
    # Y = B_TR: [y00,y01,y10,y11] = [V[2], V[3], V[6], V[7]]
    # Z = C_R: [z00,z01,z10,z11] = [W[4], W[6], W[5], W[7]]
    add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7])
    
    # Block 4: S4 = A_R * B_BR -> C_R
    # X = A_R: [x00,x01,x10,x11] = [U[2], U[3], U[6], U[7]]
    # Y = B_BR: [y00,y01,y10,y11] = [V[10], V[11], V[14], V[15]]
    # Z = C_R: [z00,z01,z10,z11] = [W[4], W[6], W[5], W[7]]
    add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        # Safety: return what we have
        pass
    
    return {"U": U, "V": V, "W": W}
