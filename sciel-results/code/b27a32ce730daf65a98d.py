import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # We split p into p1=4 and p2=3
    p1 = 4
    p2 = 3
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x3 using 4 blocks of 2x2x3 (Rank 4 * 11 = 44)
    
    rank_total = 49 + 44
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # --- Part 1: Strassen 4x4x4 ---
    # We use the Kronecker product of two 2x2x2 Strassen tensors.
    # Local indices for 2x2x2 Strassen:
    # A: 2x2, B: 2x2, C: 2x2
    # U_2x2: 4x7, V_2x2: 4x7, W_2x2: 4x7
    
    # Define Strassen 2x2x2 coefficients
    # c in 0..6
    # U_2x2[i, j, c], V_2x2[j, k, c], W_2x2[k, i, c]
    
    # M1: A11 B11 -> C11
    # M2: A12 (B21+B22) -> C12, C22
    # M3: (A11+A12) B21 -> C11, C12
    # M4: A21 (B11+B12) -> C21, C22
    # M5: (A21+A22) B11 -> C11, C21
    # M6: A22 B12 -> C22
    # M7: (A11+A22) (B22-B11) -> C11, C12, C22
    
    # Let's explicitly define the 2x2x2 Strassen decomposition
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Indices: i,j in 0..1. k in 0..1.
    # U2[i*2+j, c]
    # V2[j*2+k, c]
    # W2[k*2+i, c]
    
    # c=0: M1 = A11 B11
    U2[0, 0] = 1 # A11
    V2[0, 0] = 1 # B11
    W2[0, 0] = 1 # C11 (k=0, i=0)
    # C11 = M1 ...
    
    # c=1: M2 = A12 (B21+B22)
    U2[1, 1] = 1 # A12
    V2[2, 1] = 1 # B21
    V2[3, 1] = 1 # B22
    # C12 = ... M2 ...
    # C22 = ... M2 ...
    W2[1, 1] = 1 # C12 (k=1, i=0)
    W2[3, 1] = 1 # C22 (k=1, i=1)
    
    # c=2: M3 = (A11+A12) B21
    U2[0, 2] = 1 # A11
    U2[1, 2] = 1 # A12
    V2[2, 2] = 1 # B21
    # C11 = ... - M3 ...
    # C12 = ... + M3 ...
    W2[0, 2] = -1 # C11
    W2[1, 2] = 1  # C12
    
    # c=3: M4 = A21 (B11+B12)
    U2[2, 3] = 1 # A21
    V2[0, 3] = 1 # B11
    V2[1, 3] = 1 # B12
    # C21 = ... M4 ...
    # C22 = ... M4 ...
    W2[2, 3] = 1 # C21 (k=0, i=1)
    W2[3, 3] = 1 # C22
    
    # c=4: M5 = (A21+A22) B11
    U2[2, 4] = 1 # A21
    U2[3, 4] = 1 # A22
    V2[0, 4] = 1 # B11
    # C11 = ... + M5 ...
    # C21 = ... + M5 ...
    W2[0, 4] = 1 # C11
    W2[2, 4] = 1 # C21
    
    # c=5: M6 = A22 B12
    U2[3, 5] = 1 # A22
    V2[1, 5] = 1 # B12
    # C22 = ... M6 ...
    W2[3, 5] = 1 # C22
    
    # c=6: M7 = (A11+A22) (B22-B11)
    U2[0, 6] = 1 # A11
    U2[3, 6] = 1 # A22
    V2[3, 6] = 1 # B22
    V2[0, 6] = -1 # B11
    # C11 = ... M7 ...
    # C12 = ... M7 ...
    # C22 = ... M7 ...
    W2[0, 6] = 1 # C11
    W2[1, 6] = 1 # C12
    W2[3, 6] = 1 # C22
    
    # Verify Strassen 2x2x2
    # C11 = M1 - M3 + M5 + M7
    # C12 = M2 + M3 + M7
    # C21 = M4 + M5
    # C22 = M1 - M2 + M4 + M6 + M7
    
    # Now construct 4x4x4 using Kronecker product of (U2, V2, W2) with itself.
    # Global indices for 4x4x4:
    # i, j in 0..3. k in 0..3.
    # i = 2*i1 + i2
    # j = 2*j1 + j2
    # k = 2*k1 + k2
    
    # U_4x4[i, j, c4] where c4 = 7*c1 + c2
    # V_4x4[j, k, c4]
    # W_4x4[k, i, c4]
    
    offset_part1 = 0
    for c1 in range(7):
        for c2 in range(7):
            c4 = c1 * 7 + c2
            # U
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            val_u = U2[i1*2+i2, c1] * U2[j1*2+j2, c2]
                            if val_u != 0:
                                i_global = i1*2 + i2
                                j_global = j1*2 + j2
                                U[i_global * 4 + j_global, offset_part1 + c4] += val_u
                            
                            # V
                            val_v = V2[j1*2+j2, c1] * V2[i1*2+i2, c2] 
                            # Wait, V2 index is [j, k]. 
                            # In Kronecker, V_4x4[j, k] = V2[j1, k1] * V2[j2, k2]
                            # j = 2*j1 + j2, k = 2*k1 + k2
                            # V2 row index is j*2+k.
                            # So V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
                            
            # Let's redo the mapping more carefully.
            # U_4[i,j,c] = U2[i1,j1,c1] * U2[i2,j2,c2]
            # V_4[j,k,c] = V2[j1,k1,c1] * V2[j2,k2,c2]
            # W_4[k,i,c] = W2[k1,i1,c1] * W2[k2,i2,c2]
            
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    # This is too slow/complex to iterate fully in Python for 49*64 combos? 
                                    # 49 * 64 = 3136 iterations. Fine.
                                    
                                    i_g = i1*2 + i2
                                    j_g = j1*2 + j2
                                    k_g = k1*2 + k2
                                    
                                    u_val = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
                                    if u_val != 0:
                                        U[i_g*4 + j_g, offset_part1 + c4] += u_val
                                    
                                    v_val = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
                                    if v_val != 0:
                                        V[j_g*4 + k_g, offset_part1 + c4] += v_val
                                        
                                    w_val = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
                                    if w_val != 0:
                                        W[k_g*4 + i_g, offset_part1 + c4] += w_val

    # --- Part 2: 4x4x3 ---
    # 4 blocks of 2x2x3
    # Block (i1, j1) where i1, j1 in 0..1
    # A block: rows 2*i1..2*i1+1, cols 2*j1..2*j1+1
    # B block: rows 2*j1..2*j1+1, cols 0..2 (local)
    # C block: rows 2*i1..2*i1+1, cols 0..2 (local)
    
    offset_part2 = 49
    block_idx = 0
    for i1 in range(2):
        for j1 in range(2):
            # Construct 2x2x3 decomposition
            # Rank 11
            # 7 terms for cols 0,1 (Strassen)
            # 4 terms for col 2 (Schoolbook)
            
            # We need to map local 2x2x3 indices to global 4x4x7 indices
            # Local i in 0..1, local j in 0..1, local k in 0..2
            # Global i = 2*i1 + i
            # Global j = 2*j1 + j
            # Global k = 4 + k (since p1=4)
            
            for c_local in range(11):
                c_global = offset_part2 + block_idx * 11 + c_local
                
                if c_local < 7:
                    # Strassen part for local cols 0, 1
                    # Use U2, V2, W2 from above
                    # Local i, j in 0..1
                    for i in range(2):
                        for j in range(2):
                            u_val = U2[i*2+j, c_local]
                            if u_val != 0:
                                i_g = 2*i1 + i
                                j_g = 2*j1 + j
                                U[i_g*4 + j_g, c_global] += u_val
                            
                            # V: local j, k in 0..1
                            # V2[j*2+k, c_local]
                            for k in range(2):
                                v_val = V2[j*2+k, c_local]
                                if v_val != 0:
                                    j_g = 2*j1 + j
                                    k_g = 4 + k
                                    V[j_g*4 + k_g, c_global] += v_val
                                    
                            # W: local k, i in 0..1
                            # W2[k*2+i, c_local]
                            for k in range(2):
                                w_val = W2[k*2+i, c_local]
                                if w_val != 0:
                                    k_g = 4 + k
                                    i_g = 2*i1 + i
                                    W[k_g*4 + i_g, c_global] += w_val
                else:
                    # Schoolbook part for local col 2
                    c_sb = c_local - 7
                    # M8: A11 B13
                    # M9: A12 B23
                    # M10: A21 B13
                    # M11: A22 B23
                    if c_sb == 0:
                        U[ (2*i1+0)*4 + (2*j1+0), c_global] = 1
                        V[ (2*j1+0)*4 + (4+2), c_global] = 1
                        W[ (4+2)*4 + (2*i1+0), c_global] = 1
                    elif c_sb == 1:
                        U[ (2*i1+0)*4 + (2*j1+1), c_global] = 1
                        V[ (2*j1+1)*4 + (4+2), c_global] = 1
                        W[ (4+2)*4 + (2*i1+0), c_global] = 1
                    elif c_sb == 2:
                        U[ (2*i1+1)*4 + (2*j1+0), c_global] = 1
                        V[ (2*j1+0)*4 + (4+2), c_global] = 1
                        W[ (4+2)*4 + (2*i1+1), c_global] = 1
                    elif c_sb == 3:
                        U[ (2*i1+1)*4 + (2*j1+1), c_global] = 1
                        V[ (2*j1+1)*4 + (4+2), c_global] = 1
                        W[ (4+2)*4 + (2*i1+1), c_global] = 1
                        
            block_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
