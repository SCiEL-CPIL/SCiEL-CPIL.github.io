def construct():
    # Dimensions
    n, m, p = 4, 4, 6
    
    # Strassen 2x2x2 coefficients
    # U1[i1, r1], V1[j1, r1], W1[k1, r1]
    # i1, j1, k1 in {0, 1}
    # r1 in {0..6}
    
    U1 = [
        [1, 0, 1, 0, 1, -1, 0], # i1=0: A00
        [0, 1, 0, 1, 0, 0, -1]  # i1=1: A11
    ]
    # Wait, Strassen inputs are A00, A01, A10, A11.
    # My indices i1, i2 correspond to the 2x2 block structure.
    # In T_2,2,2, the input A is 2x2.
    # Let's map A indices (0,1) to i1, i2?
    # In the Kronecker product T_4,4,6 = T_2,2,2 (outer) x T_2,2,3 (inner).
    # The outer tensor T_2,2,2 operates on the "block" indices.
    # So i1 is the block row index, i2 is the row index within the block.
    # The U1 matrix should represent the coefficients for the block indices.
    # In Strassen, the inputs are A00, A01, A10, A11.
    # These correspond to block indices (0,0), (0,1), (1,0), (1,1).
    # So U1[i1, r1] should be the coefficient of A[i1, j1] in term r1.
    # But Strassen terms are combinations.
    # Term 1: (A00+A11)(B00+B11).
    # Coeff of A00 is 1. Coeff of A11 is 1. Coeff of A01 is 0. Coeff of A10 is 0.
    # So U1[0, 0] = 1 (A00), U1[1, 0] = 1 (A11).
    # U1[0, 1] = 0 (A01), U1[1, 1] = 1 (A11)? No.
    # Term 2: (A10+A11)B00.
    # Coeff of A10 is 1. Coeff of A11 is 1.
    # So U1[1, 1] = 1 (A10 is at i1=1, j1=0? No, A is 2x2.
    # A[i1, j1].
    # Term 1: A00 (i1=0, j1=0) and A11 (i1=1, j1=1).
    # So U1 should be 2x2? No, U1 is for the i dimension.
    # The term is rank 1 in the full tensor.
    # The factor for U is a vector of size 2 (for i1) and 2 (for j1)?
    # No, in the tensor decomposition, U is (n*m) x rank.
    # For the Kronecker product, U = U1 (n1*m1) x U2 (n2*m2)?
    # No, U is (n*m) x rank.
    # n = n1*n2. m = m1*m2.
    # U[i1*n2+i2, r1*rank2+r2] = U1[i1*m1+j1? No.
    
    # Let's re-evaluate the Kronecker structure.
    # T = T1 (n1, m1, p1) x T2 (n2, m2, p2).
    # T[i1*n2+i2, j1*m2+j2, k1*p2+k2] = T1[i1, j1, k1] * T2[i2, j2, k2].
    # This is only true if T is the Kronecker product of the tensors.
    # The matrix multiplication tensor T_MM[n,m,p] is NOT the Kronecker product of T_MM[n1,m1,p1] and T_MM[n2,m2,p2] in the sense of entry-wise multiplication of indices.
    # However, if we view the matrices as block matrices, the multiplication of blocks is the same as matrix multiplication.
    # So, if A = A1 (n1 x m1) of blocks A2 (n2 x m2), and B = B1 (m1 x p1) of blocks B2 (m2 x p2).
    # Then C = C1 (n1 x p1) of blocks C2 (n2 x p2).
    # C1[i1, j1] = sum_k1 A1[i1, k1] B1[k1, j1].
    # And each block product A1[i1, k1] B1[k1, j1] is a multiplication of n2 x m2 by m2 x p2 matrices.
    # So the total tensor is a sum of terms where each term is a product of a "block" term and a "inner" term.
    # This is exactly the Kronecker product structure if we define the indices correctly.
    
    # So, U[i, r] = U1[i1, r1] * U2[i2, r2] is NOT correct because U depends on i AND j?
    # No, U is (n*m) x rank.
    # In the Kronecker product of tensors, the U matrix is the Kronecker product of the U matrices?
    # Yes, if T = T1 x T2, then U_T = U1 x U2?
    # U1 is (n1*m1) x rank1. U2 is (n2*m2) x rank2.
    # U_T is (n1*m1*n2*m2) x (rank1*rank2).
    # U_T[(i1*m1+j1)*n2*m2 + i2*m2+j2, r1*rank2+r2] = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2].
    
    # So we need U1 for T1 (2,2,2) and U2 for T2 (2,2,3).
    # T1 is 2x2x2. n1=2, m1=2, p1=2.
    # T2 is 2x2x3. n2=2, m2=2, p2=3.
    
    # U1 size: 4 x 7.
    # U2 size: 4 x 11.
    # U size: 16 x 77.
    
    # Let's define U1, V1, W1 for Strassen.
    # Inputs A (2x2), B (2x2).
    # A indices: 0,1,2,3 (i*m+j).
    # B indices: 0,1,2,3.
    # C indices: 0,1,2,3.
    
    U1 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V1 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [1, 0, -1, 1, 0, 0, 1], # B10
        [0, 0, 0, 0, 1, 0, 1]   # B11
    ]
    
    W1 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 1]  # C11
    ]
    
    # Check Strassen:
    # P1: A00+A11, B00+B11. C00+1, C11+1.
    # U1: A00:1, A11:1. V1: B00:1, B11:1. W1: C00:1, C11:1. OK.
    # P2: A10+A11, B00. C10+1, C11-1.
    # U1: A10:1, A11:1. V1: B00:1. W1: C10:1, C11:-1. OK.
    # P3: A00, B01-B11. C01+1, C11+1.
    # U1: A00:1. V1: B01:1, B11:-1. W1: C01:1, C11:1. OK.
    # P4: A11, B10-B00. C00+1, C10+1.
    # U1: A11:1. V1: B10:1, B00:-1. W1: C00:1, C10:1. OK.
    # P5: A00+A01, B11. C00-1, C01+1.
    # U1: A00:1, A01:1. V1: B11:1. W1: C00:-1, C01:1. OK.
    # P6: A10-A00, B00+B01. C11+1.
    # U1: A10:1, A00:-1. V1: B00:1, B01:1. W1: C11:1. OK.
    # P7: A01-A11, B10+B11. C00+1, C11+1.
    # U1: A01:1, A11:-1. V1: B10:1, B11:1. W1: C00:1, C11:1. OK.
    
    # Now U2, V2, W2 for 2x2x3.
    # n2=2, m2=2, p2=3.
    # U2 size: 4 x 11.
    # V2 size: 6 x 11.
    # W2 size: 6 x 11.
    
    # Terms 0-6 are same as Strassen, but W2 has 6 entries (C00, C01, C02, C10, C11, C12).
    # C02, C12 are 0 for terms 0-6.
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0, 1, 0, 0, 0], # A00
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],  # A01
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1, 0, 0, 0, 1]  # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0, 1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # B01
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # B02
        [1, 0, -1, 1, 0, 0, 1, 0, 0, 0, 0], # B10
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # B11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]   # B12
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1, 0, 0, 0, 0], # C00
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # C01
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # C02
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 1, 0, 0, 0, 0], # C11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]   # C12
    ]
    
    # Construct final U, V, W
    rank = 7 * 11
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    
    for r1 in range(7):
        for r2 in range(11):
            r = r1 * 11 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = (i1 * 2 + i2) * 2 + (j1 * 2 + j2)
                            U[i][r] = U1[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                            
                            for k1 in range(2):
                                for k2 in range(3):
                                    k = k1 * 3 + k2
                                    # V index: j1*m2+j2 ? No, V is (m*p).
                                    # m=4, p=6.
                                    # j is 0..3, k is 0..5.
                                    # j = j1*2 + j2. k = k1*3 + k2.
                                    # V index = j*6 + k.
                                    j = j1 * 2 + j2
                                    V[j * 6 + k][r] = V1[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                    
                                    # W index: k*n+i.
                                    # k is 0..5, i is 0..3.
                                    # W index = k*4 + i.
                                    W[k * 4 + i][r] = W1[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                                    
    return {"U": U, "V": V, "W": W}
