import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 66 decomposition
    # A (3x4) split into A1 (1x4), A2 (2x4)
    # B (4x6) split into B1 (4x2), B2 (4x2), B3 (4x2)
    # C1 = A1 B1 (1x2), C2 = A1 B2 (1x2), C3 = A1 B3 (1x2)
    # C4 = A2 B1 (2x2), C5 = A2 B2 (2x2), C6 = A2 B3 (2x2)
    
    # Helper to create Strassen 2x2x2 decomposition
    # Returns U, V, W for 2x2x2 multiplication with rank 7
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j], V[i*2+j], W[i*2+j]
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[3, 6] = 1
        
        return U, V, W

    # Helper to create 1x2x2 decomposition (rank 4)
    # A (1x2) indices 0, 1
    # B (2x2) indices 0, 1, 2, 3
    # C (1x2) indices 0, 1
    def mul_122():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        U[1, 1] = 1
        V[2, 1] = 1
        W[0, 1] = 1
        
        U[0, 2] = 1
        V[1, 2] = 1
        W[1, 2] = 1
        
        U[1, 3] = 1
        V[3, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # Assemble the full decomposition
    # Total rank 66
    rank = 66
    U = np.zeros((12, rank)) # n*m = 12
    V = np.zeros((24, rank)) # m*p = 24
    W = np.zeros((18, rank)) # p*n = 18
    
    col = 0
    
    # Part 1: C1, C2, C3 (A1 * Bj)
    # A1 is rows 0 of A (indices 0, 1 in A's 1x4 block)
    # A1 = [A[0,0], A[0,1], A[0,2], A[0,3]]
    # A1a = [A[0,0], A[0,1]] -> indices 0, 1
    # A1b = [A[0,2], A[0,3]] -> indices 2, 3
    
    for j in range(3):
        # Bj is columns 2j, 2j+1 of B
        # Bj = B[0:4, 2j:2j+2]
        # Bja = B[0:2, 2j:2j+2] -> rows 0,1 of B block
        # Bjb = B[2:4, 2j:2j+2] -> rows 2,3 of B block
        
        # Term 1: A1a * Bja
        Ua, Va, Wa = mul_122()
        # Map A1a (1x2) to U indices
        # A1a indices: 0, 1
        U[0:2, col:col+4] = Ua
        # Map Bja (2x2) to V indices
        # Bja indices: 2j, 2j+1 (rows 0,1)
        # V indices for B: row r, col c -> r*p + c
        # Bja is B[0:2, 2j:2j+2]
        # Indices: 2j, 2j+1
        V[0*p + 2j : 2*p + 2j+2, col:col+4] = Va.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4) # This is tricky
        # Let's do it manually
        # V is (m*p, rank) = (24, rank)
        # Bja entries:
        # B[0, 2j], B[0, 2j+1], B[1, 2j], B[1, 2j+1]
        # Indices: 2j, 2j+1, p+2j, p+2j+1
        v_idx = [2j, 2j+1, p+2j, p+2j+1]
        for k in range(4):
            V[v_idx[k], col+k] = Va[k, k] # Va is diagonal? No, Va is (4,4)
            # Wait, mul_122 returns V as (4,4) with non-zeros on diagonal?
            # Yes, V[0,0]=1, V[2,1]=1, V[1,2]=1, V[3,3]=1
            # So V is not diagonal.
        
        # Let's just use the direct mapping
        # Va[k, l] is the coefficient for B index k in term l
        # My mul_122 V is (4, 4) where rows are B indices, cols are terms.
        # So V[v_idx[k], col + l] += Va[k, l]
        for k in range(4):
            for l in range(4):
                V[v_idx[k], col+l] = Va[k, l]
        
        # Map C (1x2) to W indices
        # C indices: 2j, 2j+1 in the j-th block of C?
        # C is 3x6.
        # C1 is row 0, cols 0,1. Indices 0, 1.
        # C2 is row 0, cols 2,3. Indices 2, 3.
        # C3 is row 0, cols 4,5. Indices 4, 5.
        # W indices: k*n + i.
        # For C1 (i=0, k=0,1): W[0*3+0]=0, W[1*3+0]=3? No.
        # W[k*n+i, r].
        # C[k, i].
        # C1 is C[0, 0], C[0, 1].
        # W indices: 0*3+0=0, 1*3+0=3.
        w_idx = [0*p + 0, 1*p + 0] # No, p=6.
        # W index is k*n + i.
        # C[0,0] -> k=0, i=0 -> 0
        # C[0,1] -> k=1, i=0 -> 3
        w_idx = [0, 3]
        for k in range(2):
            for l in range(4):
                W[w_idx[k], col+l] = Wa[k, l]
        
        col += 4
        
        # Term 2: A1b * Bjb
        Ua, Va, Wa = mul_122()
        # A1b indices: 2, 3
        U[2:4, col:col+4] = Ua
        # Bjb indices: 2j+2, 2j+3 (rows 2,3)
        v_idx = [2j+2, 2j+3, p+2j+2, p+2j+3]
        for k in range(4):
            for l in range(4):
                V[v_idx[k], col+l] = Va[k, l]
        # C indices: 0, 1 in the j-th block?
        # C1 is C[0, 2j], C[0, 2j+1].
        # W indices: 2j*3+0, (2j+1)*3+0
        w_idx = [2j*3, (2j+1)*3]
        for k in range(2):
            for l in range(4):
                W[w_idx[k], col+l] = Wa[k, l]
        
        col += 4

    # Part 2: C4, C5, C6 (A2 * Bj)
    # A2 is rows 1,2 of A.
    # A2a = A[1:3, 0:2] -> indices 4,5,6,7
    # A2b = A[1:3, 2:4] -> indices 8,9,10,11
    
    for j in range(3):
        # Term 1: A2a * Bja
        Ua, Va, Wa = strassen_222()
        # A2a indices: 4,5,6,7
        U[4:8, col:col+7] = Ua
        # Bja indices: 2j, 2j+1, p+2j, p+2j+1
        v_idx = [2j, 2j+1, p+2j, p+2j+1]
        for k in range(4):
            for l in range(7):
                V[v_idx[k], col+l] = Va[k, l]
        # C4 is C[1:3, 2j:2j+2]
        # C[1, 2j], C[1, 2j+1], C[2, 2j], C[2, 2j+1]
        # W indices:
        # C[1, 2j] -> k=2j, i=1 -> 2j*3+1
        # C[1, 2j+1] -> k=2j+1, i=1 -> (2j+1)*3+1
        # C[2, 2j] -> k=2j, i=2 -> 2j*3+2
        # C[2, 2j+1] -> k=2j+1, i=2 -> (2j+1)*3+2
        w_idx = [2j*3+1, (2j+1)*3+1, 2j*3+2, (2j+1)*3+2]
        for k in range(4):
            for l in range(7):
                W[w_idx[k], col+l] = Wa[k, l]
        
        col += 7
        
        # Term 2: A2b * Bjb
        Ua, Va, Wa = strassen_222()
        # A2b indices: 8,9,10,11
        U[8:12, col:col+7] = Ua
        # Bjb indices: 2j+2, 2j+3, p+2j+2, p+2j+3
        v_idx = [2j+2, 2j+3, p+2j+2, p+2j+3]
        for k in range(4):
            for l in range(7):
                V[v_idx[k], col+l] = Va[k, l]
        # C indices same as above
        w_idx = [2j*3+1, (2j+1)*3+1, 2j*3+2, (2j+1)*3+2]
        for k in range(4):
            for l in range(7):
                W[w_idx[k], col+l] = Wa[k, l]
        
        col += 7

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

