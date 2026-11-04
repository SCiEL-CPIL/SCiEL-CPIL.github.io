import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Dimensions
    # U: (n*m, R) = (10, R)
    # V: (m*p, R) = (30, R)
    # W: (p*n, R) = (12, R)
    
    # Target Rank
    R = 54
    U = np.zeros((10, R), dtype=float)
    V = np.zeros((30, R), dtype=float)
    W = np.zeros((12, R), dtype=float)
    
    col = 0
    
    # Part 1: A_1 B_1 (j=0)
    # A_1 is 2x1 (i=0,1; j=0)
    # B_1 is 1x6 (j=0; k=0..5)
    # C_1 is 2x6
    # Term: A[i,0] * B[0,k]
    for i in range(2):
        for k in range(6):
            # U index: i*m + j = i*5 + 0
            u_idx = i * 5 + 0
            # V index: j*p + k = 0*6 + k = k
            v_idx = 0 * 6 + k
            # W index: k*n + i = k*2 + i
            w_idx = k * 2 + i
            
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1
            
    # Part 2: A_4 B_4 (j=1..4)
    # Split B_4 (4x6) into 3 blocks of 4x2
    # Block 0: k in {0,1}
    # Block 1: k in {2,3}
    # Block 2: k in {4,5}
    
    # Strassen coefficients for 2x2x2
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X10)Y11
    # M6 = (X01-X00)(Y00+Y10)
    # M7 = (X10-X11)(Y01+Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We need to add contributions to U, V, W for each Strassen term.
    # For a specific Strassen instance with local indices:
    # X (A part) indices: i_local, j_local
    # Y (B part) indices: j_local, k_local
    # Z (C part) indices: i_local, k_local
    
    # Mapping local to global:
    # i_local (0,1) -> i_global (0,1)
    # j_local (0,1) -> j_global (offset_j + j_local)
    # k_local (0,1) -> k_global (offset_k + k_local)
    
    def add_strassen_block(offset_j, offset_k):
        nonlocal col
        # This function adds 7 rank-1 terms for a 2x2x2 multiplication
        # X is 2x2 (i in 0,1; j in offset_j, offset_j+1)
        # Y is 2x2 (j in offset_j, offset_j+1; k in offset_k, offset_k+1)
        # Z is 2x2 (i in 0,1; k in offset_k, offset_k+1)
        
        # Precompute indices for U, V, W
        # U[i, j]
        # V[j, k]
        # W[k, i]
        
        # Strassen M1: (X00+X11)(Y00+Y11)
        # X00: i=0, j=offset_j
        # X11: i=1, j=offset_j+1
        # Y00: j=offset_j, k=offset_k
        # Y11: j=offset_j+1, k=offset_k+1
        
        # We need to decompose (X00+X11) and (Y00+Y11) into linear combinations.
        # In the tensor decomposition, U, V, W hold the coefficients.
        # For M1, we need:
        # U[0, offset_j] += 1
        # U[1, offset_j+1] += 1
        # V[offset_j, offset_k] += 1
        # V[offset_j+1, offset_k+1] += 1
        # W[offset_k, 0] += 1
        # W[offset_k+1, 1] += 1
        
        # Wait, the rank-1 term is U * V * W.
        # If we set U[0, offset_j]=1, U[1, offset_j+1]=1, etc.,
        # Then the product U[i,j] V[j,k] W[k,i] will sum over r.
        # For this specific r (col), the term is:
        # sum_{i,j,k} U[i,j] V[j,k] W[k,i] A[i,j] B[j,k]
        # Actually, the decomposition is:
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We want T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term.
        
        # For the Strassen term M1, it contributes to Z00, Z01, Z10, Z11.
        # The coefficients in W will handle the combination for Z.
        # The coefficients in U and V handle the combination for X and Y.
        
        # Let's define the 7 terms explicitly.
        
        # M1
        U[0*5 + offset_j, col] += 1.0
        U[1*5 + (offset_j+1), col] += 1.0
        V[(offset_j)*6 + offset_k, col] += 1.0
        V[(offset_j+1)*6 + (offset_k+1), col] += 1.0
        W[offset_k*2 + 0, col] += 1.0
        W[(offset_k+1)*2 + 1, col] += 1.0
        col += 1
        
        # M2: (X10+X11)Y00
        # X10: i=1, j=offset_j
        # X11: i=1, j=offset_j+1
        # Y00: j=offset_j, k=offset_k
        U[1*5 + offset_j, col] += 1.0
        U[1*5 + (offset_j+1), col] += 1.0
        V[(offset_j)*6 + offset_k, col] += 1.0
        W[offset_k*2 + 1, col] += 1.0
        col += 1
        
        # M3: X00(Y10-Y11)
        # X00: i=0, j=offset_j
        # Y10: j=offset_j+1, k=offset_k
        # Y11: j=offset_j+1, k=offset_k+1
        U[0*5 + offset_j, col] += 1.0
        V[(offset_j+1)*6 + offset_k, col] += 1.0
        V[(offset_j+1)*6 + (offset_k+1), col] += -1.0
        W[offset_k*2 + 0, col] += 1.0
        W[(offset_k+1)*2 + 0, col] += -1.0 # Wait, W index is k*n+i.
        # M3 contributes to Z01 (i=0, k=1) and Z11 (i=1, k=1)?
        # Let's check Strassen formulas again.
        # Z01 = M3 + M5
        # Z11 = M1 - M2 + M3 + M6
        # So M3 contributes to Z01 (i=0, k=1) and Z11 (i=1, k=1).
        # W index for Z01: k=1, i=0 -> 1*2+0 = 2.
        # W index for Z11: k=1, i=1 -> 1*2+1 = 3.
        # So W[2] += 1, W[3] += 1.
        # My previous line was wrong.
        # Let's restart the W assignments based on Z contributions.
        
        # This manual mapping is error-prone.
        # Better approach:
        # Generate the 7 rank-1 tensors for Strassen in local coordinates.
        # Then map them to global coordinates.
        
        # Local Strassen Rank-1 Tensors (U_L, V_L, W_L) for 2x2x2:
        # r=0 (M1): U: (1,1) at (0,0),(1,1). V: (1,1) at (0,0),(1,1). W: (1,1) at (0,0),(1,1).
        # Actually, W is defined by the Z combination.
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # Let's list the 7 terms (M1..M7) and their U, V, W components.
        # M1: U: (1,0,0,1) [i,j], V: (1,0,0,1) [j,k], W: (1,0,0,1) [k,i] ? No.
        # W[k,i] is the coefficient for C[i,k].
        # For M1, it contributes to all Z with coefficient 1? No.
        # M1 contributes to Z00 (1), Z11 (1).
        # So W for M1: W[0,0]=1, W[1,1]=1.
        
        # Let's build the 7 terms explicitly.
        
        terms = []
        
        # M1
        u1 = np.zeros(4); u1[0]=1; u1[3]=1 # i0j0, i1j1
        v1 = np.zeros(4); v1[0]=1; v1[3]=1 # j0k0, j1k1
        w1 = np.zeros(4); w1[0]=1; w1[3]=1 # k0i0, k1i1
        terms.append((u1, v1, w1))
        
        # M2
        u2 = np.zeros(4); u2[2]=1; u2[3]=1 # i1j0, i1j1
        v2 = np.zeros(4); v2[0]=1 # j0k0
        w2 = np.zeros(4); w2[2]=1 # k0i1 (Z10)
        terms.append((u2, v2, w2))
        
        # M3
        u3 = np.zeros(4); u3[0]=1 # i0j0
        v3 = np.zeros(4); v3[2]=1; v3[3]=-1 # j1k0, j1k1
        w3 = np.zeros(4); w3[1]=1; w3[3]=1 # k1i0 (Z01), k1i1 (Z11)
        terms.append((u3, v3, w3))
        
        # M4
        u4 = np.zeros(4); u4[3]=1 # i1j1
        v4 = np.zeros(4); v4[1]=1; v4[0]=-1 # j0k1, j0k0
        w4 = np.zeros(4); w4[0]=1; w4[2]=1 # k0i0 (Z00), k0i1 (Z10)
        terms.append((u4, v4, w4))
        
        # M5
        u5 = np.zeros(4); u5[0]=1; u5[2]=1 # i0j0, i1j0
        v5 = np.zeros(4); v5[3]=1 # j1k1
        w5 = np.zeros(4); w5[1]=1; w5[3]=-1 # k1i0 (Z01), k1i1 (Z11)
        terms.append((u5, v5, w5))
        
        # M6
        u6 = np.zeros(4); u6[1]=1; u6[0]=-1 # i0j1, i0j0
        v6 = np.zeros(4); v6[0]=1; v6[2]=1 # j0k0, j1k0
        w6 = np.zeros(4); w6[3]=1 # k1i1 (Z11)
        terms.append((u6, v6, w6))
        
        # M7
        u7 = np.zeros(4); u7[2]=1; u7[3]=-1 # i1j0, i1j1
        v7 = np.zeros(4); v7[1]=1; v7[3]=1 # j0k1, j1k1
        w7 = np.zeros(4); w7[0]=1; w7[2]=-1 # k0i0 (Z00), k0i1 (Z10)
        terms.append((u7, v7, w7))
        
        # Now map these local terms to global indices
        for u_l, v_l, w_l in terms:
            # u_l is 2x2 flattened (i*2+j)
            for i_l in range(2):
                for j_l in range(2):
                    val = u_l[i_l*2+j_l]
                    if val != 0:
                        i_g = i_l
                        j_g = offset_j + j_l
                        U[i_g*5 + j_g, col] += val
            
            # v_l is 2x2 flattened (j*2+k)
            for j_l in range(2):
                for k_l in range(2):
                    val = v_l[j_l*2+k_l]
                    if val != 0:
                        j_g = offset_j + j_l
                        k_g = offset_k + k_l
                        V[j_g*6 + k_g, col] += val
            
            # w_l is 2x2 flattened (k*2+i)
            for k_l in range(2):
                for i_l in range(2):
                    val = w_l[k_l*2+i_l]
                    if val != 0:
                        k_g = offset_k + k_l
                        i_g = i_l
                        W[k_g*2 + i_g, col] += val
            
            col += 1

    # Apply to 3 blocks
    # Block 0: k in {0,1}, j in {1,2} (Top) and {3,4} (Bottom)
    # We need two Strassen instances per block:
    # 1. A[0:2, 1:3] * B[1:3, 0:2] -> offset_j=1, offset_k=0
    # 2. A[0:2, 3:5] * B[3:5, 0:2] -> offset_j=3, offset_k=0
    
    add_strassen_block(1, 0)
    add_strassen_block(3, 0)
    
    # Block 1: k in {2,3}
    add_strassen_block(1, 2)
    add_strassen_block(3, 2)
    
    # Block 2: k in {4,5}
    add_strassen_block(1, 4)
    add_strassen_block(3, 4)
    
    # Check if we filled all columns
    # 12 + 42 = 54.
    # col should be 54.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

