import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition (rank 7)
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = -1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[3, 6] = -1
    
    # Trivial 3x3 decomposition (rank 27)
    R3 = 27
    U3 = np.zeros((9, R3), dtype=float)
    V3 = np.zeros((9, R3), dtype=float)
    W3 = np.zeros((9, R3), dtype=float)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3 + j, col] = 1
                V3[j*3 + k, col] = 1
                W3[k*3 + i, col] = 1
                col += 1
    
    # Kronecker product: 6x6 from 2x2 and 3x3
    # Rank = 7 * 27 = 189
    R6 = 7 * 27
    U6 = np.zeros((36, R6), dtype=float)
    V6 = np.zeros((36, R6), dtype=float)
    W6 = np.zeros((36, R6), dtype=float)
    
    for r2 in range(7):
        for r3 in range(27):
            r6 = r2 * 27 + r3
            # U6: index (i2*3+i3, j2*3+j3) for 6x6, from U2[i2,j2] * U3[i3,j3]
            # Actually, U2 has shape (4,7) where row index is i*2+j for 2x2
            # U3 has shape (9,27) where row index is i*3+j for 3x3
            # For 6x6, row index is i*6+j where i = i2*3+i3, j = j2*3+j3
            for i2 in range(2):
                for j2 in range(2):
                    for i3 in range(3):
                        for j3 in range(3):
                            i6 = i2 * 3 + i3
                            j6 = j2 * 3 + j3
                            U6[i6 * 6 + j6, r6] = U2[i2 * 2 + j2, r2] * U3[i3 * 3 + j3, r3]
                            V6[j6 * 6 + (j2 * 3 + j3 if False else 0), r6] = 0  # placeholder
            # This is getting complex. Let me use a cleaner approach.
    
    # Actually, the Kronecker product of tensor decompositions works as follows:
    # If (U1,V1,W1) is a rank-r1 decomposition of n1xn1xn1 and
    # (U2,V2,W2) is a rank-r2 decomposition of n2xn2xn2,
    # then the Kronecker product gives a rank-r1*r2 decomposition of (n1*n2)x(n1*n2)x(n1*n2)
    # where U_kr[i1*n2+i2, j1*n2+j2, r1*r2+r2] = U1[i1,j1,r1] * U2[i2,j2,r2]
    # Wait, no. The indices are:
    # U_kr[(i1*n2+i2)*(n1*n2) + (j1*n2+j2), r1*r2+r2] 
    # Hmm, this is getting confusing. Let me think more carefully.
    
    # For matrix multiplication tensor: T[i,j,k] = delta[i,k] * A[i,j] * B[j,k]
    # The decomposition is: T = sum_r U[:,r] V[:,r] W[:,r]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # U[i*m+j, r] corresponds to A[i,j]
    # V[j*p+k, r] corresponds to B[j,k]  
    # W[k*n+i, r] corresponds to C[i,k]
    
    # For Kronecker product of 2x2 and 3x3:
    # 6x6 matrix: A6[i6,j6] where i6 = i2*3+i3, j6 = j2*3+j3
    # A6[i6,j6] = A2[i2,j2] * A3[i3,j3] (Kronecker product of matrices)
    # But we want to decompose the multiplication tensor, not the matrices.
    
    # The multiplication tensor for 6x6: T6[i6,j6,k6]
    # = T2[i2,j2,k2] * T3[i3,j3,k3] where i6=i2*3+i3, j6=j2*3+j3, k6=k2*3+k3
    # This is the Kronecker product of the tensors.
    
    # So: U6[i6*6+j6, r6] = U2[i2*2+j2, r2] * U3[i3*3+j3, r3]
    #     V6[j6*6+k6, r6] = V2[j2*2+k2, r2] * V3[j3*3+k3, r3]
    #     W6[k6*6+i6, r6] = W2[k2*2+i2, r2] * W3[k3*3+i3, r3]
    # where r6 = r2*27 + r3
    
    U6 = np.zeros((36, 189), dtype=float)
    V6 = np.zeros((36, 189), dtype=float)
    W6 = np.zeros((36, 189), dtype=float)
    
    for r2 in range(7):
        for r3 in range(27):
            r6 = r2 * 27 + r3
            for i2 in range(2):
                for j2 in range(2):
                    for k2 in range(2):
                        for i3 in range(3):
                            for j3 in range(3):
                                for k3 in range(3):
                                    i6 = i2 * 3 + i3
                                    j6 = j2 * 3 + j3
                                    k6 = k2 * 3 + k3
                                    
                                    u_val = U2[i2 * 2 + j2, r2] * U3[i3 * 3 + j3, r3]
                                    v_val = V2[j2 * 2 + k2, r2] * V3[j3 * 3 + k3, r3]
                                    w_val = W2[k2 * 2 + i2, r2] * W3[k3 * 3 + i3, r3]
                                    
                                    U6[i6 * 6 + j6, r6] += u_val
                                    V6[j6 * 6 + k6, r6] += v_val
                                    W6[k6 * 6 + i6, r6] += w_val
    
    # Now extract 5x5 from 6x6 by keeping only indices 0-4 for i,j,k
    # U5: shape (25, 189), U5[i*5+j, r] = U6[i*6+j, r] for i,j in 0..4
    # V5: shape (25, 189), V5[j*5+k, r] = V6[j*6+k, r] for j,k in 0..4
    # W5: shape (25, 189), W5[k*5+i, r] = W6[k*6+i, r] for k,i in 0..4
    
    U5 = np.zeros((25, 189), dtype=float)
    V5 = np.zeros((25, 189), dtype=float)
    W5 = np.zeros((25, 189), dtype=float)
    
    for i in range(5):
        for j in range(5):
            U5[i * 5 + j, :] = U6[i * 6 + j, :]
    for j in range(5):
        for k in range(5):
            V5[j * 5 + k, :] = V6[j * 6 + k, :]
    for k in range(5):
        for i in range(5):
            W5[k * 5 + i, :] = W6[k * 6 + i, :]
    
    # Convert to lists
    U = U5.tolist()
    V = V5.tolist()
    W = W5.tolist()
    
    return {"U": U, "V": V, "W": W}
