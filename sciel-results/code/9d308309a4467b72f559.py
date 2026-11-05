import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2x2
    def get_strassen_2x2x2():
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices for 2x2: 0,1 for rows/cols of A, 0,1 for rows of B (cols of A), 0,1 for cols of B
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[1,0] = 1
        W[0,0] = 1; W[3,0] = 1; W[1,0] = 1; W[2,0] = 1 # C11, C22, C12, C21? No, W maps to output C.
        # Strassen W coefficients:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[1,0]=1
        W[0,0]=1; W[3,0]=1 # C11, C22? No, W[k*n+i, c]. Output index is (i,k).
        # Output C is 2x2. Indices (0,0), (0,1), (1,0), (1,1).
        # W[0,0] corresponds to C[0,0]? No, W is (p*n, rank). p=2, n=2. Size 4.
        # Index in W: k*n + i.
        # C[0,0] -> 0*2+0=0.
        # C[0,1] -> 1*2+0=2.
        # C[1,0] -> 0*2+1=1.
        # C[1,1] -> 1*2+1=3.
        
        # M1: C11 += 1, C22 += 1.
        W[0,0] = 1
        W[3,0] = 1
        
        # M2 = (A21+A22)B11
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[1,1] = 1 # C21
        W[3,1] = -1 # C22
        
        # M3 = A11(B12-B22)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[2,2] = 1 # C12
        W[3,2] = 1 # C22
        
        # M4 = A22(B21-B11)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3] = 1 # C11
        W[1,3] = 1 # C21
        
        # M5 = (A11+A12)B22
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4] = -1 # C11
        W[2,4] = 1 # C12
        
        # M6 = (A21-A11)(B11+B12)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5] = 1 # C22
        
        # M7 = (A12-A22)(B21+B22)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6] = 1 # C11
        W[2,6] = 1 # C12
        
        return U, V, W

    # 3x3x3 decomposition
    def get_3x3x3():
        U_s, V_s, W_s = get_strassen_2x2x2()
        # We need to map 2x2 indices to 3x3 indices.
        # A indices: 0,1,2. B indices: 0,1,2. C indices: 0,1,2.
        # 2x2 block uses 0,1.
        # a is A[:, 2] (col 2). b is A[2, :] (row 2).
        # c is B[:, 2] (col 2). b33 is B[2, 2].
        
        U = np.zeros((9, 20))
        V = np.zeros((9, 20))
        W = np.zeros((9, 20))
        
        # 1. A22 B22 (Rank 7)
        # Map U_s (4x7) to U (9x7) at indices 0,1.
        U[0:2, 0:7] = U_s
        V[0:2, 0:7] = V_s
        W[0:2*2+0:2*2+2, 0:7] = W_s # W indices for C[0:2, 0:2] are 0,1,2,3.
        
        # 2. a c^T (Rank 1)
        # a = A[0:2, 2]. c = B[0:2, 2].
        # Term: A[0:2, 2] * B[2, 0:2]? No.
        # Partition: A = [[A22, a], [bT, a33]].
        # A[0:2, 2] is a. A[2, 0:2] is bT.
        # B = [[B22, b], [cT, b33]].
        # B[0:2, 2] is b. B[2, 0:2] is cT.
        # Term a c^T: A[0:2, 2] * B[2, 0:2].
        # C[0:2, 0:2] += a * cT.
        # U[0:2, 7] = 1 (for a)
        # V[2, 7] = 1 (for cT? No, V index is j. j=2)
        # W[0:2*2+0:2*2+2, 7] = 1
        U[0,7] = 1; U[1,7] = 1
        V[2,7] = 1
        W[0,7] = 1; W[1,7] = 1; W[2,7] = 1; W[3,7] = 1
        
        # 3. A22 b (Rank 4)
        # C[0:2, 2] += A22 * b.
        # b = B[0:2, 2].
        # This is 2x2 matrix times 2x1 vector.
        # 4 terms.
        # Term 3.1: A[0,0] B[0,2]
        U[0,8] = 1; V[0,8] = 1; W[2,8] = 1 # C[0,2]
        # Term 3.2: A[0,1] B[1,2]
        U[1,9] = 1; V[1,9] = 1; W[2,9] = 1 # C[0,2]
        # Term 3.3: A[1,0] B[0,2]
        U[2,10] = 1; V[0,10] = 1; W[4,10] = 1 # C[1,2]
        # Term 3.4: A[1,1] B[1,2]
        U[3,11] = 1; V[1,11] = 1; W[4,11] = 1 # C[1,2]
        
        # 4. a b33 (Rank 1)
        # C[0:2, 2] += a * b33.
        U[0,12] = 1; U[1,12] = 1
        V[2,12] = 1 # b33 is B[2,2]
        W[2,12] = 1; W[4,12] = 1
        
        # 5. bT B22 (Rank 4)
        # C[2, 0:2] += bT * B22.
        # bT = A[2, 0:2].
        # Term 5.1: A[2,0] B[0,0]
        U[6,13] = 1; V[0,13] = 1; W[0,13] = 1 # C[2,0]
        # Term 5.2: A[2,0] B[1,0]
        U[6,14] = 1; V[1,14] = 1; W[0,14] = 1 # C[2,0]
        # Term 5.3: A[2,1] B[0,1]
        U[7,15] = 1; V[0,15] = 1; W[2,15] = 1 # C[2,1]
        # Term 5.4: A[2,1] B[1,1]
        U[7,16] = 1; V[1,16] = 1; W[2,16] = 1 # C[2,1]
        
        # 6. b33 cT (Rank 1)
        # C[2, 0:2] += b33 * cT.
        U[8,17] = 1 # b33 is A[2,2]
        V[2,17] = 1 # cT is B[2, 0:2]
        W[0,17] = 1; W[2,17] = 1
        
        # 7. bT b (Rank 1)
        # C[2,2] += bT * b.
        # bT = A[2, 0:2], b = B[0:2, 2].
        # Dot product.
        # U[6,18]=1, V[0,18]=1, W[6,18]=1
        # U[7,18]=1, V[1,18]=1, W[6,18]=1
        # Wait, this sums to 2 terms.
        U[6,18] = 1; V[0,18] = 1; W[6,18] = 1
        U[7,19] = 1; V[1,19] = 1; W[6,19] = 1
        # Rank 2?
        # Yes, dot product of length 2 is rank 2.
        # So total rank 20+1 = 21?
        # My previous count: 7+1+4+1+4+1+1+1 = 20.
        # bT b is rank 2.
        # So 21.
        
        return U, V, W

    # 3x3x2 decomposition
    def get_3x3x2():
        U_s, V_s, W_s = get_strassen_2x2x2()
        U = np.zeros((9, 13))
        V = np.zeros((6, 13)) # B is 3x2. Size 6.
        W = np.zeros((6, 13)) # C is 3x2. Size 6.
        
        # 1. A22 B22 (Rank 7)
        U[0:2, 0:7] = U_s
        V[0:2, 0:7] = V_s
        W[0:2*2+0:2*2+2, 0:7] = W_s
        
        # 2. a b32 (Rank 1)
        # a = A[0:2, 2]. b32 = B[2, 0:2].
        U[0,7] = 1; U[1,7] = 1
        V[4,7] = 1; V[5,7] = 1 # B[2,0], B[2,1]
        W[0,7] = 1; W[1,7] = 1; W[2,7] = 1; W[3,7] = 1
        
        # 3. bT B22 (Rank 4)
        # C[2, 0:2] += bT * B22.
        U[6,8] = 1; V[0,8] = 1; W[4,8] = 1
        U[6,9] = 1; V[1,9] = 1; W[4,9] = 1
        U[7,10] = 1; V[0,10] = 1; W[5,10] = 1
        U[7,11] = 1; V[1,11] = 1; W[5,11] = 1
        
        # 4. a33 b32 (Rank 1)
        # C[2, 0:2] += a33 * b32.
        U[8,12] = 1
        V[4,12] = 1; V[5,12] = 1
        W[4,12] = 1; W[5,12] = 1
        
        return U, V, W

    U1, V1, W1 = get_3x3x3()
    U2, V2, W2 = get_3x3x2()
    
    U = np.concatenate([U1, U2], axis=1)
    V = np.concatenate([V1, V2], axis=1)
    W = np.concatenate([W1, W2], axis=1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
