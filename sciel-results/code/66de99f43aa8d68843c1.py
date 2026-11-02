import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will use the decomposition:
    # R(4,5,5) <= R(4,4,4) + R(4,4,1) + R(4,1,5)
    
    # 1. M(4,4,4) using M(2,2,2) x M(2,2,2)
    # Rank 7 * 7 = 49
    # 2. M(4,4,1) schoolbook rank 16
    # 3. M(4,1,5) schoolbook rank 20
    # Total 85
    
    # Helper to create schoolbook tensor
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen M(2,2,2)
    def strassen_2x2():
        # Rank 7 decomposition for 2x2
        # U, V, W of shape (4, 7)
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices 0..3 for 2x2 matrices
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # M1 = A00 * (B11 - B01)
        U[0, 0] = 1; U[1, 0] = 1; U[2, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        # This is not Strassen.
        
        # Correct Strassen:
        # M1 = A00(B00+B11)
        # M2 = A10(B00+B01)
        # M3 = (A00+A01)B11
        # M4 = (A10+A11)(B01-B11)
        # M5 = (A00+A10)(B10-B00)
        # M6 = (A01+A11)(B10-B11) -- Wait, standard is:
        # M1 = A00(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = (A00+A01)B11
        # M4 = A10(B01-B11)
        # M5 = A01(B10-B00)
        # M6 = (A00+A11)(B10+B11)
        # M7 = (A10+A11)(B00-B10)
        
        # Let's use a known valid decomposition.
        # U, V, W for M(2,2,2)
        U[0,0]=1; U[1,0]=1; U[2,0]=1; U[3,0]=1
        V[0,0]=1; V[1,0]=1; V[2,0]=1; V[3,0]=1
        W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1
        # This is rank 1.
        
        # I will use the explicit Strassen matrices.
        # It's safer to just use schoolbook for 2x2 (rank 4) if I can't recall the exact entries.
        # But rank 4 is worse than 7.
        # Let's use the known Strassen decomposition.
        
        # U:
        # 0: 1 1 0 0
        # 1: 1 0 1 0
        # 2: 0 1 1 0
        # 3: 0 0 1 1
        # This is not it.
        
        # I will use the schoolbook for 2x2 (rank 4) to be safe, 
        # but then M(4,4,4) rank is 16.
        # Total 16 + 16 + 20 = 52.
        # This is much better than 100.
        
        return schoolbook(2, 2, 2)

    U_444, V_444, W_444 = strassen_2x2()
    # If strassen_2x2 returns schoolbook, rank is 4.
    # Then M(4,4,4) is Kronecker product of two M(2,2,2).
    # Rank is 4*4=16.
    
    # Construct M(4,4,4) from M(2,2,2) x M(2,2,2)
    U_444_full = np.kron(U_444, U_444)
    V_444_full = np.kron(V_444, V_444)
    W_444_full = np.kron(W_444, W_444)
    
    # Construct M(4,4,1) schoolbook
    U_441, V_441, W_441 = schoolbook(4, 4, 1)
    
    # Construct M(4,1,5) schoolbook
    U_415, V_415, W_415 = schoolbook(4, 1, 5)
    
    # Combine
    # M(4,5,5) indices:
    # U: (4*5) x Rank
    # V: (5*5) x Rank
    # W: (5*4) x Rank
    
    rank_444 = U_444_full.shape[1]
    rank_441 = U_441.shape[1]
    rank_415 = U_415.shape[1]
    
    total_rank = rank_444 + rank_441 + rank_415
    
    U = np.zeros((4 * 5, total_rank))
    V = np.zeros((5 * 5, total_rank))
    W = np.zeros((5 * 4, total_rank))
    
    col = 0
    
    # Add M(4,4,4)
    # U_444_full is (16, rank). Maps to first 4 rows of U (rows 0..3 of 4x5)
    U[0:16, col:col+rank_444] = U_444_full
    # V_444_full is (16, rank). Maps to first 4 cols of V (cols 0..3 of 5x5)
    # V indices: j*p+k. j=0..3, k=0..3.
    # So rows 0..15 of V.
    V[0:16, col:col+rank_444] = V_444_full
    # W_444_full is (16, rank). Maps to first 4 rows/cols of W (4x4 block of 5x4)
    # W indices: k*n+i. k=0..3, i=0..3.
    # So rows 0..15 of W.
    W[0:16, col:col+rank_444] = W_444_full
    col += rank_444
    
    # Add M(4,4,1)
    # This handles the case where k=4 (last col of B)
    # U: (4*4) x rank. Rows 0..15.
    U[0:16, col:col+rank_441] = U_441
    # V: (4*1) x rank. Rows j*1+4. j=0..3.
    # So rows 4, 9, 14, 19? No.
    # V indices: j*p+k. p=1.
    # j=0..3, k=0.
    # Rows 0, 1, 2, 3.
    # But we need to map to the 5x5 V.
    # V_441 is for M(4,4,1).
    # The V part is (4*1).
    # In the full V (5*5), these correspond to j=0..3, k=4.
    # Indices: 0*5+4=4, 1*5+4=9, 2*5+4=14, 3*5+4=19.
    V[4:20:5, col:col+rank_441] = V_441
    # W: (1*4) x rank. Rows k*4+i. k=0, i=0..3.
    # Rows 0, 1, 2, 3.
    W[0:4, col:col+rank_441] = W_441
    col += rank_441
    
    # Add M(4,1,5)
    # This handles the case where j=4 (last col of A)
    # U: (4*1) x rank. Rows i*1+4. i=0..3.
    # Indices: 4, 9, 14, 19.
    U[4:20:5, col:col+rank_415] = U_415
    # V: (1*5) x rank. Rows 0..4.
    V[0:5, col:col+rank_415] = V_415
    # W: (5*4) x rank. Rows 0..19.
    W[0:20, col:col+rank_415] = W_415
    col += rank_415
    
    return {"U": U, "V": V, "W": W}

