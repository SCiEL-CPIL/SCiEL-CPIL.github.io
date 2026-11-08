import numpy as np

def construct():
    n, m, p = 3, 4, 4
    N = n * m  # 12
    P = m * p  # 16
    Q = p * n  # 12
    
    # Rank 9 decomposition for M(3,2,2)
    # A is 3x2, B is 2x2, C is 3x2
    # U is 6x9, V is 4x9, W is 6x9
    
    U_loc = np.zeros((6, 9))
    V_loc = np.zeros((4, 9))
    W_loc = np.zeros((6, 9))
    
    # Part 1: M(2,2,2) for rows 0,1 of A (indices 0,1 in U_loc, 0,1 in W_loc)
    # Strassen rank 7
    # A = [[a,b],[c,d]], B = [[x,y],[z,w]], C = [[c00,c01],[c10,c11]]
    # U indices: 0=a, 1=b, 2=c, 3=d
    # V indices: 0=x, 1=y, 2=z, 3=w
    # W indices: 0=c00, 1=c01, 2=c10, 3=c11
    
    # M1 = (a+c)(x+z)
    U_loc[0, 0] = 1; U_loc[2, 0] = 1
    V_loc[0, 0] = 1; V_loc[2, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1 # Wait, Strassen output mapping
    # Standard Strassen:
    # M1 = (a+c)(x+z) -> contributes to c00, c11
    # M2 = (c+d)(x+z) -> contributes to c10, c11
    # M3 = (a-b)(x+y) -> contributes to c00, c01
    # M4 = (b-d)(y+w) -> contributes to c10, c11
    # M5 = (a+d)(x+w) -> contributes to c00, c11
    # M6 = (a+c-b-d)(x+y-z-w) -> contributes to c00, c01, c10, c11
    # M7 = (b+c-a-d)(x+z-y-w) -> contributes to c00, c01, c10, c11
    
    # Let's use a known explicit Strassen implementation.
    # Reference: 
    # M1 = (a+c)(x+z)
    # M2 = (c+d)(x+z)
    # M3 = (a-b)(x+y)
    # M4 = (b-d)(y+w)
    # M5 = (a+d)(x+w)
    # M6 = (a+c-b-d)(x+y-z-w)
    # M7 = (b+c-a-d)(x+z-y-w)
    # C00 = M1 + M3 - M5 + M6
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M4 + M7
    
    # Col 0: M1
    U_loc[0, 0] = 1; U_loc[2, 0] = 1
    V_loc[0, 0] = 1; V_loc[2, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # Col 1: M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1; V_loc[2, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = 1
    
    # Col 2: M3
    U_loc[0, 2] = 1; U_loc[1, 2] = -1
    V_loc[0, 2] = 1; V_loc[1, 2] = 1
    W_loc[0, 2] = 1; W_loc[1, 2] = 1
    
    # Col 3: M4
    U_loc[1, 3] = 1; U_loc[3, 3] = -1
    V_loc[1, 3] = 1; V_loc[3, 3] = 1
    W_loc[2, 3] = 1; W_loc[3, 3] = 1
    
    # Col 4: M5
    U_loc[0, 4] = 1; U_loc[3, 4] = 1
    V_loc[0, 4] = 1; V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[3, 4] = 1
    
    # Col 5: M6
    U_loc[0, 5] = 1; U_loc[2, 5] = 1; U_loc[1, 5] = -1; U_loc[3, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1; V_loc[2, 5] = -1; V_loc[3, 5] = -1
    W_loc[0, 5] = 1; W_loc[1, 5] = 1; W_loc[2, 5] = 1; W_loc[3, 5] = 1
    
    # Col 6: M7
    U_loc[1, 6] = 1; U_loc[2, 6] = 1; U_loc[0, 6] = -1; U_loc[3, 6] = -1
    V_loc[0, 6] = 1; V_loc[2, 6] = 1; V_loc[1, 6] = -1; V_loc[3, 6] = -1
    W_loc[0, 6] = 1; W_loc[1, 6] = 1; W_loc[2, 6] = 1; W_loc[3, 6] = 1
    
    # Part 2: M(1,2,2) for row 2 of A (index 4,5 in U_loc, 4,5 in W_loc)
    # Rank 2
    # U indices: 4=e, 5=f
    # V indices: 0=x, 1=y, 2=z, 3=w
    # W indices: 4=c20, 5=c21
    
    # Col 7: e*x + f*z -> c20
    U_loc[4, 7] = 1; U_loc[5, 7] = 1
    V_loc[0, 7] = 1; V_loc[2, 7] = 1
    W_loc[4, 7] = 1
    
    # Col 8: e*y + f*w -> c21
    U_loc[4, 8] = 1; U_loc[5, 8] = 1
    V_loc[1, 8] = 1; V_loc[3, 8] = 1
    W_loc[5, 8] = 1
    
    # Assemble global U, V, W
    U = np.zeros((N, 36))
    V = np.zeros((P, 36))
    W = np.zeros((Q, 36))
    
    # Blocks:
    # B is 4x4. Split into 2x2 blocks.
    # B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3)
    # B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # A is 3x4. Split into 3x2 blocks.
    # AL (cols 0-1), AR (cols 2-3)
    # C is 3x4. Split into 3x2 blocks.
    # CL (cols 0-1), CR (cols 2-3)
    
    # C_L = A_L B_00 + A_R B_10
    # C_R = A_L B_01 + A_R B_11
    
    # Mapping of local indices to global indices
    # A: i (0..2), j (0..1) -> ga = i*4 + j (if AL) or i*4 + 2 + j (if AR)
    # B: i (0..1), j (0..1) -> gb = (br+i)*4 + (bc+j)
    # C: i (0..2), j (0..1) -> gc = (cc+j)*3 + (cr+i)
    
    blocks = [
        # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
        (0, 0, 0, 0), # AL, B00 -> CL
        (2, 2, 0, 0), # AR, B10 -> CL
        (0, 0, 2, 2), # AL, B01 -> CR
        (2, 2, 2, 2), # AR, B11 -> CR
    ]
    
    for b_idx, (ac, br, bc, cc) in enumerate(blocks):
        offset = b_idx * 9
        
        # A indices
        for i in range(3):
            for j in range(2):
                la = i * 2 + j
                ga = i * 4 + (ac + j)
                U[ga, offset:offset+9] += U_loc[la, :]
                
        # B indices
        for i in range(2):
            for j in range(2):
                lb = i * 2 + j
                gb = (br + i) * 4 + (bc + j)
                V[gb, offset:offset+9] += V_loc[lb, :]
                
        # C indices
        for i in range(3):
            for j in range(2):
                lc = i * 2 + j
                gc = (cc + j) * 3 + (cr + i) # Wait, C is 3x4. Index is k*n+i?
                # Task says W[k*n+i, r].
                # C is 3x4. Rows are k (0..3)? No, C is 3x4.
                # A is 3x4, B is 4x4. C is 3x4.
                # C_{i,k} = sum_j A_{i,j} B_{j,k}.
                # W index is k*n + i.
                # Here C is split into 3x2 blocks.
                # C_L is cols 0,1. C_R is cols 2,3.
                # In C_L, col index is 0 or 1.
                # In C_R, col index is 2 or 3.
                # gc = k*3 + i.
                # k is the column index of C.
                # In the block, j is the local column index (0 or 1).
                # Global column index k = cc + j.
                # So gc = (cc + j) * 3 + i.
                # My previous loop used (cc+j)*3 + (cr+i). But cr is always 0 here?
                # No, C rows are 0..2. There is no row split for C in this block decomposition.
                # The "cr" in the previous code was confusing.
                # C is not split into row blocks, only column blocks.
                # So i is the global row index 0..2.
                # So gc = (cc + j) * 3 + i.
                
    # Re-implementing the loop correctly
    U = np.zeros((N, 36))
    V = np.zeros((P, 36))
    W = np.zeros((Q, 36))
    
    for b_idx, (ac, br, bc, cc) in enumerate(blocks):
        offset = b_idx * 9
        
        for i in range(3):
            for j in range(2):
                la = i * 2 + j
                ga = i * 4 + (ac + j)
                U[ga, offset:offset+9] += U_loc[la, :]
                
        for i in range(2):
            for j in range(2):
                lb = i * 2 + j
                gb = (br + i) * 4 + (bc + j)
                V[gb, offset:offset+9] += V_loc[lb, :]
                
        for i in range(3):
            for j in range(2):
                lc = i * 2 + j
                gc = (cc + j) * 3 + i
                W[gc, offset:offset+9] += W_loc[lc, :]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
