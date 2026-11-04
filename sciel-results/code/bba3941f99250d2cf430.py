import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get rank 7 decomposition for 2x2x2
    def get_2x2_2x2_2x2():
        # Strassen rank 7
        # U, V, W are 4x7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00 at (0,0), A11 at (1,1)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C00 += M1, C01 += M1? No, M1 is 2x2.
        # C = [[C00, C01], [C10, C11]]
        # M1 contributes to all? No.
        # Strassen reconstruction:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map the 7 multiplications to the final C.
        # Each multiplication m produces a 2x2 matrix M_m.
        # The factors U, V, W for the 2x2x2 tensor are such that:
        # sum_r U[i,r] V[j,r] W[k,r] = delta_{ij,k}? No.
        # The tensor is T_{i,j,k} = 1 if (i,j,k) is a valid term.
        # For 2x2x2, T_{i,j,k} = 1 for all i,j,k? No.
        # T_{i,j,k} = 1 if the term A_ij B_jk contributes to C_ik.
        # So T_{i,j,k} = 1 for all i,j,k in {0,1}.
        
        # The rank 7 decomposition of the 2x2x2 tensor:
        # U, V, W are 4x7.
        # We can use the standard Strassen factors.
        
        # Step 0: M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        U[0,0]=1; U[3,0]=1
        # V: B00=1, B11=1
        V[0,0]=1; V[3,0]=1
        # W: C00+=1, C01+=1, C10+=1, C11+=1? No.
        # M1 is added to C00 and C11?
        # C00 = M1 - M4 + M5 - M7
        # C11 = M1 - M2 + M3 + M6
        # So W for M1: W[0,0]=1 (C00), W[3,0]=1 (C11)
        W[0,0]=1; W[3,0]=1
        
        # Step 1: M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        # C10 += M2, C11 -= M2
        W[2,1]=1; W[3,1]=-1
        
        # Step 2: M3 = A00(B01-B11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        # C01 += M3, C11 += M3
        W[1,2]=1; W[3,2]=1
        
        # Step 3: M4 = A11(B10-B00)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        # C00 -= M4, C10 += M4
        W[0,3]=-1; W[2,3]=1
        
        # Step 4: M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        # C00 += M5, C01 += M5
        W[0,4]=1; W[1,4]=1
        
        # Step 5: M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        # C11 += M6
        W[3,5]=1
        
        # Step 6: M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        # C00 -= M7
        W[0,6]=-1
        
        return U, V, W

    U22, V22, W22 = get_2x2_2x2_2x2()
    
    # Construct 4x4x4 rank 49
    # 7 Strassen steps, each 2x2x2 rank 7
    U1 = np.zeros((16, 49))
    V1 = np.zeros((16, 49))
    W1 = np.zeros((16, 49))
    
    # Block indices: 00, 01, 10, 11
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # Mapping from block (bi, bj) to index bi*4+bj?
    # A index: i*4+j. Block (bi,bj) covers i in {2bi, 2bi+1}, j in {2bj, 2bj+1}.
    
    # For each Strassen step s (0..6), and each 2x2 factor f (0..6):
    # col = s*7 + f
    
    for s in range(7):
        for f in range(7):
            col = s * 7 + f
            
            # Determine A block coefficients from U22[:, f]
            # U22 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # A00: U22[0,f], A01: U22[1,f], A10: U22[2,f], A11: U22[3,f]
            
            # Map to 4x4 U1
            # A00 is at rows 0,1 and cols 0,1 -> indices 0,1,4,5
            # A01 is at rows 0,1 and cols 2,3 -> indices 2,3,6,7
            # A10 is at rows 2,3 and cols 0,1 -> indices 8,9,12,13
            # A11 is at rows 2,3 and cols 2,3 -> indices 10,11,14,15
            
            if U22[0, f] != 0:
                U1[[0,1,4,5], col] = U22[0, f]
            if U22[1, f] != 0:
                U1[[2,3,6,7], col] = U22[1, f]
            if U22[2, f] != 0:
                U1[[8,9,12,13], col] = U22[2, f]
            if U22[3, f] != 0:
                U1[[10,11,14,15], col] = U22[3, f]
                
            # Determine B block coefficients from V22[:, f]
            # V22 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # B00: V22[0,f], B01: V22[1,f], B10: V22[2,f], B11: V22[3,f]
            
            # B index: j*4+k.
            # B00: j in {0,1}, k in {0,1} -> 0,1,4,5
            # B01: j in {0,1}, k in {2,3} -> 2,3,6,7
            # B10: j in {2,3}, k in {0,1} -> 8,9,12,13
            # B11: j in {2,3}, k in {2,3} -> 10,11,14,15
            
            if V22[0, f] != 0:
                V1[[0,1,4,5], col] = V22[0, f]
            if V22[1, f] != 0:
                V1[[2,3,6,7], col] = V22[1, f]
            if V22[2, f] != 0:
                V1[[8,9,12,13], col] = V22[2, f]
            if V22[3, f] != 0:
                V1[[10,11,14,15], col] = V22[3, f]
                
            # Determine C block coefficients from W22[:, f]
            # W22 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # C00: W22[0,f], C01: W22[1,f], C10: W22[2,f], C11: W22[3,f]
            
            # C index: k*4+i.
            # C00: k in {0,1}, i in {0,1} -> 0,1,4,5
            # C01: k in {0,1}, i in {2,3} -> 2,3,6,7
            # C10: k in {2,3}, i in {0,1} -> 8,9,12,13
            # C11: k in {2,3}, i in {2,3} -> 10,11,14,15
            
            if W22[0, f] != 0:
                W1[[0,1,4,5], col] = W22[0, f]
            if W22[1, f] != 0:
                W1[[2,3,6,7], col] = W22[1, f]
            if W22[2, f] != 0:
                W1[[8,9,12,13], col] = W22[2, f]
            if W22[3, f] != 0:
                W1[[10,11,14,15], col] = W22[3, f]
                
    # Construct 4x4x2 rank 28
    # 4 multiplications of 2x2x2
    # M1 = A11 B1 (A00, B00? No, A11 is bottom-left? No, A11 is bottom-right block)
    # Let's use standard block notation:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]] -- Wait, B is 4x2.
    # B = [[B_top], [B_bottom]] where B_top is 2x2, B_bottom is 2x2.
    # Let B1 = B_top, B2 = B_bottom.
    # C = A B = [[A00, A01], [A10, A11]] [[B1], [B2]]
    # C = [[A00 B1 + A01 B2], [A10 B1 + A11 B2]]
    # C_top = A00 B1 + A01 B2
    # C_bottom = A10 B1 + A11 B2
    
    # We need 4 multiplications:
    # P1 = A00 B1
    # P2 = A01 B2
    # P3 = A10 B1
    # P4 = A11 B2
    
    U2 = np.zeros((16, 28))
    V2 = np.zeros((8, 28)) # 4x2 -> 8 elements
    W2 = np.zeros((8, 28)) # 4x2 -> 8 elements
    
    # B index: j*2 + k' where k' in {0,1}.
    # B1 (top): j in {0,1}, k' in {0,1} -> 0,1,2,3
    # B2 (bottom): j in {2,3}, k' in {0,1} -> 4,5,6,7
    
    # C index: k*4 + i.
    # C_top: k in {0,1}, i in {0,3} -> 0..7
    # C_bottom: k in {2,3}, i in {0,3} -> 8..15? No, C is 4x2.
    # C index: k*4 + i.
    # k=0,1 -> C_top rows.
    # k=2,3 -> C_bottom rows.
    
    # W2 shape (8, 28). Rows correspond to k*4+i for k in {0,1}?
    # Wait, W2 is for the 4x4x2 part.
    # The full W has rows k*4+i for k in {0..5}.
    # For the second part, k in {4,5}.
    # Let k' = k - 4. k' in {0,1}.
    # W2 row index: k'*4 + i.
    
    # Multiplication 1: A00 B1
    # A00: rows 0,1 cols 0,1 -> 0,1,4,5
    # B1: rows 0,1 cols 0,1 -> 0,1,2,3
    # C_top_left? No, C_top is 2x2.
    # C_top = A00 B1 + A01 B2.
    # A00 B1 contributes to C_top.
    # C_top indices: k in {0,1}, i in {0,1} (for A00 part)?
    # A00 is 2x2. B1 is 2x2. Product is 2x2.
    # This product goes to C[0:2, 0:2].
    # C indices: k=0,1; i=0,1.
    # W2 indices: k'*4+i. k'=0,1. i=0,1.
    # So W2 rows 0,1,4,5.
    
    for f in range(7):
        col = f
        if U22[0, f] != 0: U2[[0,1,4,5], col] = U22[0, f]
        if V22[0, f] != 0: V2[[0,1,2,3], col] = V22[0, f]
        if W22[0, f] != 0: W2[[0,1,4,5], col] = W22[0, f]
        
    # Multiplication 2: A01 B2
    # A01: rows 0,1 cols 2,3 -> 2,3,6,7
    # B2: rows 2,3 cols 0,1 -> 4,5,6,7
    # Product goes to C[0:2, 2:4]? No, B is 4x2.
    # B2 is bottom 2x2 of B.
    # A01 is top-right 2x2 of A.
    # A01 B2 is 2x2.
    # It contributes to C_top?
    # C_top = A00 B1 + A01 B2.
    # Yes, C_top is 2x2.
    # So it contributes to C[0:2, 0:2]?
    # Wait, C is 4x2.
    # C_top is rows 0,1.
    # C_bottom is rows 2,3.
    # A00 B1 + A01 B2 = C[0:2, :].
    # So both P1 and P2 contribute to C[0:2, :].
    # P1 (A00 B1) -> C[0:2, 0:2]? No, B1 is cols 0,1 of B?
    # B is 4x2. B1 is rows 0,1 of B.
    # So B1 is 2x2.
    # A00 is 2x2.
    # A00 B1 is 2x2.
    # This result is added to C[0:2, :].
    # Specifically, C[0:2, 0:2] = A00 B1?
    # No, C[0:2, :] = A[0:2, :] B[:, :].
    # A[0:2, :] = [A00, A01].
    # B[:, :] = [[B1], [B2]].
    # C[0:2, :] = A00 B1 + A01 B2.
    # So P1 and P2 are added to the same C block (top rows).
    # W2 for P1: W22[0,f] -> W2[0,1,4,5] (k'=0,1, i=0,1).
    # W2 for P2: W22[0,f] -> W2[0,1,4,5] (k'=0,1, i=0,1).
    
    for f in range(7):
        col = 7 + f
        if U22[1, f] != 0: U2[[2,3,6,7], col] = U22[1, f]
        if V22[1, f] != 0: V2[[4,5,6,7], col] = V22[1, f]
        if W22[0, f] != 0: W2[[0,1,4,5], col] = W22[0, f] # Same C block
        
    # Multiplication 3: A10 B1
    # A10: rows 2,3 cols 0,1 -> 8,9,12,13
    # B1: rows 0,1 cols 0,1 -> 0,1,2,3
    # Product goes to C[2:4, :].
    # W2 indices: k'=2,3? No, k' in {0,1} for W2.
    # W2 row index: k'*4+i.
    # C_bottom corresponds to k in {2,3} in full C.
    # In W2, k' = k - 4? No, W2 is for the 4x4x2 part.
    # The 4x4x2 part uses k in {0,1} for the local index?
    # No, the 4x4x2 part is for columns 4,5 of B.
    # So k in {4,5}.
    # Let k_local = k - 4. k_local in {0,1}.
    # C_local is 4x2.
    # C_local[0:2, :] = C_top_local.
    # C_local[2:4, :] = C_bottom_local.
    # A10 B1 contributes to C_local[2:4, :].
    # So W2 rows: k_local in {0,1}? No.
    # C_local index: k_local*4 + i.
    # C_local[2:4, :] means k_local in {0,1}? No.
    # C_local is 4 rows.
    # Rows 0,1 are top. Rows 2,3 are bottom.
    # So k_local is 0,1 for top? No.
    # C_local[i, k_local].
    # i is row 0..3. k_local is col 0..1.
    # W2 row index: k_local*4 + i.
    # For C_local[2:4, :], i in {2,3}.
    # So W2 rows: k_local*4 + i.
    # k_local=0 -> 0,1,2,3. i=2,3 -> 2,3.
    # k_local=1 -> 4,5,6,7. i=2,3 -> 6,7.
    # So W2 rows 2,3,6,7.
    
    for f in range(7):
        col = 14 + f
        if U22[2, f] != 0: U2[[8,9,12,13], col] = U22[2, f]
        if V22[0, f] != 0: V2[[0,1,2,3], col] = V22[0, f]
        if W22[0, f] != 0: W2[[2,3,6,7], col] = W22[0, f]
        
    # Multiplication 4: A11 B2
    # A11: rows 2,3 cols 2,3 -> 10,11,14,15
    # B2: rows 2,3 cols 0,1 -> 4,5,6,7
    # Product goes to C_local[2:4, :].
    # W2 rows 2,3,6,7.
    
    for f in range(7):
        col = 21 + f
        if U22[3, f] != 0: U2[[10,11,14,15], col] = U22[3, f]
        if V22[1, f] != 0: V2[[4,5,6,7], col] = V22[1, f]
        if W22[0, f] != 0: W2[[2,3,6,7], col] = W22[0, f]
        
    # Combine
    U = np.hstack([U1, U2])
    
    # V1 is 16x49. V2 is 8x28.
    # Full V is 24x77.
    # V rows: j*6+k.
    # For k<4 (Part 1): j in 0..3, k in 0..3.
    # V1 rows: j*4+k.
    # Map V1[j*4+k] to V[j*6+k].
    
    V = np.zeros((24, 77))
    for j in range(4):
        for k in range(4):
            v1_idx = j * 4 + k
            v_full_idx = j * 6 + k
            V[v_full_idx, :49] = V1[v1_idx, :]
            
    # For k>=4 (Part 2): j in 0..3, k in 4..5.
    # k' = k - 4.
    # V2 rows: j*2+k'.
    # Map V2[j*2+k'] to V[j*6+k].
    
    for j in range(4):
        for kp in range(2):
            k = kp + 4
            v2_idx = j * 2 + kp
            v_full_idx = j * 6 + k
            V[v_full_idx, 49:77] = V2[v2_idx, :]
            
    # W1 is 16x49. W2 is 8x28.
    # Full W is 24x77.
    # W rows: k*4+i.
    # For k<4 (Part 1): k in 0..3, i in 0..3.
    # W1 rows: k*4+i.
    # Map W1[k*4+i] to W[k*4+i].
    
    W = np.zeros((24, 77))
    for k in range(4):
        for i in range(4):
            w1_idx = k * 4 + i
            w_full_idx = k * 4 + i
            W[w_full_idx, :49] = W1[w1_idx, :]
            
    # For k>=4 (Part 2): k in 4..5, i in 0..3.
    # k' = k - 4.
    # W2 rows: k'*4+i.
    # Map W2[k'*4+i] to W[k*4+i].
    
    for kp in range(2):
        k = kp + 4
        for i in range(4):
            w2_idx = kp * 4 + i
            w_full_idx = k * 4 + i
            W[w_full_idx, 49:77] = W2[w2_idx, :]
            
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
