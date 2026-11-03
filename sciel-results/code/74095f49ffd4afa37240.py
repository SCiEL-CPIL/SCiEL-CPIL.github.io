import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 2, 2, 1
    # Split p=6 into 2, 2, 2
    # Rank of 2x2 by 2x2 is 7 (Strassen)
    # Rank of 2x1 by 1x6 is 12 (Schoolbook)
    # Total rank = 7*3 + 7*3 + 12 = 21 + 21 + 12 = 54
    
    # We will build the decomposition explicitly.
    
    # Helper to get Strassen decomposition for 2x2 by 2x2
    # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 multiplications
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # Mapping to indices:
        # A11 -> U[0], A12 -> U[1], A21 -> U[2], A22 -> U[3]
        # B11 -> V[0], B12 -> V[1], B21 -> V[2], B22 -> V[3]
        # C11 -> W[0], C12 -> W[1], C21 -> W[2], C22 -> W[3]
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
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
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    U_s, V_s, W_s = strassen_2x2()
    
    # We need to assemble the full U, V, W for 2x5 by 5x6
    # U shape: (2*5, rank) = (10, rank)
    # V shape: (5*6, rank) = (30, rank)
    # W shape: (6*2, rank) = (12, rank)
    
    rank = 54
    U = np.zeros((10, rank))
    V = np.zeros((30, rank))
    W = np.zeros((12, rank))
    
    col = 0
    # Part 1: A1 (2x2) * B1 (2x6)
    # B1 is rows 0-1 of B, cols 0-5
    # Split B1 into B1_1 (2x2), B1_2 (2x2), B1_3 (2x2)
    for part in range(3):
        # Use Strassen for this 2x2 by 2x2
        for r in range(7):
            # U indices: A1 is rows 0-1 of A, cols 0-1
            # U row index = i*m + j. i in 0..1, j in 0..1.
            # So U[0..3]
            U[0:4, col] = U_s[:, r]
            
            # V indices: B1_part is rows 0-1 of B, cols part*2 .. part*2+1
            # V row index = j*p + k. j in 0..1, k in part*2 .. part*2+1.
            # V[0:4] corresponds to j=0,1 and k=0,1.
            # We need to map V_s indices to V.
            # V_s[0] -> j=0, k=0 -> V[0]
            # V_s[1] -> j=0, k=1 -> V[1]
            # V_s[2] -> j=1, k=0 -> V[2]
            # V_s[3] -> j=1, k=1 -> V[3]
            
            # But V is 30x7.
            # V row = j*6 + k.
            # For j=0, k=0 -> 0
            # For j=0, k=1 -> 1
            # For j=1, k=0 -> 6
            # For j=1, k=1 -> 7
            
            v_indices = [0, 1, 6, 7]
            if part == 1:
                v_indices = [2, 3, 8, 9]
            elif part == 2:
                v_indices = [4, 5, 10, 11]
                
            for i in range(4):
                V[v_indices[i], col] = V_s[i, r]
                
            # W indices: C part is cols part*2 .. part*2+1 of C
            # W row index = k*n + i. k in 0..5, i in 0..1.
            # W_s[0] -> k=0, i=0 -> W[0]
            # W_s[1] -> k=0, i=1 -> W[1]
            # W_s[2] -> k=1, i=0 -> W[2]
            # W_s[3] -> k=1, i=1 -> W[3]
            
            w_indices = [0, 1, 2, 3]
            if part == 1:
                w_indices = [4, 5, 6, 7]
            elif part == 2:
                w_indices = [8, 9, 10, 11]
                
            for i in range(4):
                W[w_indices[i], col] = W_s[i, r]
                
            col += 1
            
    # Part 2: A2 (2x2) * B2 (2x6)
    # A2 is cols 2-3 of A.
    # B2 is rows 2-3 of B.
    for part in range(3):
        for r in range(7):
            # U indices: A2 is rows 0-1 of A, cols 2-3
            # U row index = i*5 + j.
            # i=0, j=2 -> 2
            # i=0, j=3 -> 3
            # i=1, j=2 -> 7
            # i=1, j=3 -> 8
            u_indices = [2, 3, 7, 8]
            for i in range(4):
                U[u_indices[i], col] = U_s[i, r]
                
            # V indices: B2_part is rows 2-3 of B, cols part*2 .. part*2+1
            # V row index = j*6 + k.
            # j=2, k=0 -> 12
            # j=2, k=1 -> 13
            # j=3, k=0 -> 18
            # j=3, k=1 -> 19
            v_indices = [12, 13, 18, 19]
            if part == 1:
                v_indices = [14, 15, 20, 21]
            elif part == 2:
                v_indices = [16, 17, 22, 23]
                
            for i in range(4):
                V[v_indices[i], col] = V_s[i, r]
                
            # W indices: Same as before
            w_indices = [0, 1, 2, 3]
            if part == 1:
                w_indices = [4, 5, 6, 7]
            elif part == 2:
                w_indices = [8, 9, 10, 11]
                
            for i in range(4):
                W[w_indices[i], col] = W_s[i, r]
                
            col += 1

    # Part 3: A3 (2x1) * B3 (1x6)
    # A3 is col 4 of A.
    # B3 is row 4 of B.
    # Rank 12.
    for i in range(2):
        for k in range(6):
            U[i*5 + 4, col] = 1
            V[4*6 + k, col] = 1
            W[k*2 + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
