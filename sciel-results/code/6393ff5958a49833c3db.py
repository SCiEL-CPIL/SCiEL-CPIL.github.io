import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    
    def get_base_221():
        # 2x2 x 2x1
        U = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V = [
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]
        W = [
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]
        return U, V, W

    def get_strassen_222():
        # 2x2 x 2x2
        U = [
            [1, 0, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 1, 0]
        ]
        V = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 0]
        ]
        W = [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0]
        ]
        return U, V, W

    def combine_decomps(d1, d2, k1, k2):
        # d1: U1, V1, W1 for k1
        # d2: U2, V2, W2 for k2
        # Result for k1+k2
        U1, V1, W1 = d1
        U2, V2, W2 = d2
        
        U = U1 + U2
        
        # V1 is 2xk1, V2 is 2xk2
        # We need to embed V1 in 2x(k1+k2) at start, V2 at end
        V = [[0]*(k1+k2) for _ in range(2)]
        for i in range(2):
            for j in range(k1):
                V[i][j] = V1[i][j]
            for j in range(k2):
                V[i][k1+j] = V2[i][j]
                
        # W1 is 2xk1, W2 is 2xk2
        W = [[0]*(k1+k2) for _ in range(2)]
        for i in range(2):
            for j in range(k1):
                W[i][j] = W1[i][j]
            for j in range(k2):
                W[i][k1+j] = W2[i][j]
                
        return U, V, W

    # Build M_22_5
    d_221 = get_base_221()
    d_222 = get_strassen_222()
    d_223 = combine_decomps(d_222, d_221, 2, 1)
    d_225 = combine_decomps(d_222, d_223, 2, 3)
    
    # Now combine two M_22_5 for M_24_5
    # A = [A_L, A_R], B = [B_T; B_B]
    # C = A_L B_T + A_R B_B
    # U is 8x18 for first, 8x18 for second
    # V is 10x18 for first, 10x18 for second
    # W is 10x18 for first, 10x18 for second
    
    U1, V1, W1 = d_225
    U2, V2, W2 = d_225
    
    # U: 8x36
    U = [[0]*36 for _ in range(8)]
    for i in range(4):
        for j in range(18):
            U[i][j] = U1[i][j]
            U[i+4][j+18] = U2[i][j]
            
    # V: 10x36
    # V1 is 2x5 (B_T), V2 is 2x5 (B_B)
    # Global V is 4x5 (B). Rows 0,1 are B_T, Rows 2,3 are B_B.
    # V1 indices: row 0,1.
    # V2 indices: row 2,3.
    V = [[0]*36 for _ in range(10)]
    for i in range(2):
        for j in range(5):
            V[i*5+j][0:18] = [V1[i][j]] * 1 # No, V1[i][j] is a value
            # V1[i][j] is the coefficient for term j?
            # No, V1 is a matrix. V1[i][j] is the entry at row i, col j.
            # But we are constructing the V matrix of the decomposition.
            # The decomposition has 18 terms.
            # For term t, V1[i][t] is the value.
            # So V[i*5+j][t] = V1[i][t] for i in 0,1.
            pass
    
    # Correct mapping:
    # V is (m*p) x rank = 10 x 36.
    # V1 is 2x18. V1[i][t] for i in 0,1, t in 0..17.
    # V2 is 2x18. V2[i][t] for i in 0,1, t in 0..17.
    # Global B index: row r, col c. Index r*5+c.
    # For V1 (B_T), r is 0,1. So global row is r.
    # For V2 (B_B), r is 2,3. So global row is r+2.
    
    for t in range(18):
        for i in range(2):
            for c in range(5):
                V[i*5+c][t] = V1[i][c] # Wait, V1[i][c]?
                # V1 is 2x5 matrix? No, V1 is 2x18 matrix in decomp.
                # In get_strassen, V is 2x4? No, V is 2x4 for k=2.
                # In combine, V is 2x(k1+k2).
                # So V1 is 2x5. V1[i][t] is the value for term t.
                # Yes.
                
    for t in range(18):
        for i in range(2):
            for c in range(5):
                V[(i+2)*5+c][t+18] = V2[i][c]
                
    # W: 10x36
    # W1 is 2x5. W1[i][t].
    # Global C index: row i, col c. Index i*5+c?
    # No, W is (p*n) x rank. p=5, n=2.
    # W[k*n+i, t]. k is col of C, i is row of C.
    # In our subproblem, C is 2x5. i in 0,1, k in 0..4.
    # Index k*2+i.
    
    for t in range(18):
        for i in range(2):
            for k in range(5):
                W[k*2+i][t] = W1[i][k]
                
    for t in range(18):
        for i in range(2):
            for k in range(5):
                W[k*2+i][t+18] = W2[i][k]
                
    return {"U": U, "V": V, "W": W}
