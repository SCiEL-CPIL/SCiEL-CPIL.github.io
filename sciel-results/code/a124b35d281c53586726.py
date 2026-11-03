import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Decomposition:
    # A = [[A_L, a], [b^T, c]]
    # B = [[B_L], [b_vec]]
    # C = A_L B_L + a b_vec^T + b^T B_L + c b_vec^T
    
    # 1. A_L B_L: 2x2x6 using Strassen (rank 21)
    # 2. a b_vec^T: 2x1x6 (rank 12)
    # 3. b^T B_L: 1x2x6 (rank 12)
    # 4. c b_vec^T: 1x1x6 (rank 6)
    
    rank = 21 + 12 + 12 + 6
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # 1. Strassen for 2x2x6
    # B_L is rows 0,1 of B. C_L is rows 0,1 of C.
    # A_L is rows 0,1 and cols 0,1 of A.
    # Strassen for 2x2x2 applied to 3 column pairs.
    for k_pair in range(3):
        k1 = k_pair * 2
        k2 = k_pair * 2 + 1
        # M1 = (A00+A11)(B0k1+B1k2) ... wait, B is 2x6.
        # B_L[k1, :] and B_L[k2, :]? No, B_L is 2x6.
        # Columns are k=0..5.
        # We group columns (0,1), (2,3), (4,5).
        # For each pair, we do 2x2x2 multiplication.
        
        # A_L indices: (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3 in U
        # B_L indices: (0,0)..(0,5), (1,0)..(1,5) -> 0..5, 6..11 in V
        # C_L indices: (0,0)..(0,5), (1,0)..(1,5) -> 0..5, 6..11 in W
        
        # Strassen 7 terms
        # 1. (A00+A11)(B0k1+B1k1) -> C0k1 += M1, C1k1 += M1, C0k2 += M1, C1k2 += M1? No.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 - M3 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M4
        # C22 = M1 - M2 + M4 + M6
        
        # We need to map this to our indices.
        # A_L[0,0]=U[0], A_L[0,1]=U[1], A_L[1,0]=U[2], A_L[1,1]=U[3]
        # B_L[0,k1]=V[k1], B_L[0,k2]=V[k2]
        # B_L[1,k1]=V[6+k1], B_L[1,k2]=V[6+k2]
        # C_L[0,k1]=W[k1], C_L[0,k2]=W[k2]
        # C_L[1,k1]=W[6+k1], C_L[1,k2]=W[6+k2]
        
        # Term 1: (U0+U3)(V[k1]+V[6+k1])
        U[col, 0] = 1; U[col, 3] = 1
        V[col, k1] = 1; V[col, 6+k1] = 1
        W[col, k1] = 1; W[col, 6+k1] = 1
        # Wait, C11 and C22 get M1. C12 and C21 get -M1?
        # C11 = M1 - M3 + M5 - M7
        # C22 = M1 - M2 + M4 + M6
        # So W[k1] (C00) gets +1. W[6+k2] (C11) gets +1.
        # But M1 is (A11+A22)(B11+B22).
        # This contributes to C11 and C22.
        # In our 0-indexed A_L: A11=U[0], A22=U[3].
        # B11=V[k1], B22=V[6+k2].
        # So M1 = (U[0]+U[3])(V[k1]+V[6+k2]).
        # C00 (W[k1]) gets M1? No, C11 in Strassen notation is C[0,0].
        # C[0,0] = M1 - M3 + M5 - M7.
        # C[1,1] = M1 - M2 + M4 + M6.
        # So W[k1] += 1, W[6+k2] += 1.
        # Wait, I used V[k1] and V[6+k1] in the draft. B11 is B[0,0] of the 2x2 block.
        # The 2x2 block of B is columns k1, k2.
        # So B11 = B[0,k1], B12 = B[0,k2], B21 = B[1,k1], B22 = B[1,k2].
        # M1 = (A11+A22)(B11+B22) = (U[0]+U[3])(V[k1]+V[6+k2]).
        # C11 (W[k1]) gets +M1.
        # C22 (W[6+k2]) gets +M1.
        
        # Let's correct the indices.
        U[col, 0] = 1; U[col, 3] = 1
        V[col, k1] = 1; V[col, 6+k2] = 1
        W[col, k1] = 1; W[col, 6+k2] = 1
        col += 1
        
        # M2 = (A21+A22)B11 = (U[2]+U[3])V[k1]
        # C21 (W[6+k1]) gets +M2.
        U[col, 2] = 1; U[col, 3] = 1
        V[col, k1] = 1
        W[col, 6+k1] = 1
        col += 1
        
        # M3 = A11(B12-B22) = U[0](V[k2]-V[6+k2])
        # C11 (W[k1]) gets -M3. C12 (W[k2]) gets +M3.
        U[col, 0] = 1
        V[col, k2] = 1; V[col, 6+k2] = -1
        W[col, k1] = -1; W[col, k2] = 1
        col += 1
        
        # M4 = A22(B21-B11) = U[3](V[6+k1]-V[k1])
        # C21 (W[6+k1]) gets +M4. C22 (W[6+k2]) gets +M4.
        U[col, 3] = 1
        V[col, 6+k1] = 1; V[col, k1] = -1
        W[col, 6+k1] = 1; W[col, 6+k2] = 1
        col += 1
        
        # M5 = (A11+A12)B22 = (U[0]+U[1])V[6+k2]
        # C11 (W[k1]) gets +M5. C22 (W[6+k2]) gets -M5.
        U[col, 0] = 1; U[col, 1] = 1
        V[col, 6+k2] = 1
        W[col, k1] = 1; W[col, 6+k2] = -1
        col += 1
        
        # M6 = (A21-A11)(B11+B12) = (U[2]-U[0])(V[k1]+V[k2])
        # C22 (W[6+k2]) gets +M6.
        U[col, 2] = 1; U[col, 0] = -1
        V[col, k1] = 1; V[col, k2] = 1
        W[col, 6+k2] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22) = (U[1]-U[3])(V[6+k1]+V[6+k2])
        # C11 (W[k1]) gets -M7. C12 (W[k2]) gets +M7.
        U[col, 1] = 1; U[col, 3] = -1
        V[col, 6+k1] = 1; V[col, 6+k2] = 1
        W[col, k1] = -1; W[col, k2] = 1
        col += 1

    # 2. a b_vec^T: 2x1x6
    # a is A[0,2], A[1,2] -> U[4], U[5]
    # b_vec is B[2,0..5] -> V[12..17]
    # C is C[0,0..5], C[1,0..5] -> W[0..5], W[6..11]
    for k in range(6):
        U[col, 4] = 1
        V[col, 12+k] = 1
        W[col, k] = 1
        col += 1
        U[col, 5] = 1
        V[col, 12+k] = 1
        W[col, 6+k] = 1
        col += 1

    # 3. b^T B_L: 1x2x6
    # b is A[2,0], A[2,1] -> U[6], U[7]
    # B_L is B[0,0..5], B[1,0..5] -> V[0..5], V[6..11]
    # C is C[2,0..5] -> W[12..17]
    for k in range(6):
        U[col, 6] = 1
        V[col, k] = 1
        W[col, 12+k] = 1
        col += 1
        U[col, 7] = 1
        V[col, 6+k] = 1
        W[col, 12+k] = 1
        col += 1

    # 4. c b_vec^T: 1x1x6
    # c is A[2,2] -> U[8]
    # b_vec is B[2,0..5] -> V[12..17]
    # C is C[2,0..5] -> W[12..17]
    for k in range(6):
        U[col, 8] = 1
        V[col, 12+k] = 1
        W[col, 12+k] = 1
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
