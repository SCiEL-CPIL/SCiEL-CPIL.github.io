import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We construct the decomposition for 3x3x3 using block decomposition (2+1 split)
    # and then duplicate it for the 6 dimension (3+3 split).
    
    def get_2x2x2_factors():
        # Strassen decomposition for 2x2x2
        # Rank 7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10 + A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00(B10 - B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[2, 2] = 1
        
        # M4 = A11(B01 - B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[1, 3] = 1; W[3, 3] = 1
        
        # M5 = (A00 + A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[2, 5] = 1; W[0, 5] = 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    def get_simple_factors(n, m, p):
        # Schoolbook for small dimensions
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

    def get_3x3x3_factors():
        # Block decomposition: 3 = 2 + 1
        # A = [[A22, A21], [A12, A11]]
        # B = [[B22, B21], [B12, B11]]
        # C = [[C22, C21], [C12, C11]]
        
        # We will build U, V, W of size (9, R), (9, R), (9, R)
        # But we need to map indices correctly.
        
        # Let's collect all blocks
        blocks = []
        
        # C22 = A22 B22 + A21 B12
        # A22: i in 0,1; j in 0,1. B22: j in 0,1; k in 0,1.
        # A21: i in 0,1; j in 2. B12: j in 2; k in 0,1.
        
        U22, V22, W22 = get_2x2x2_factors()
        # Map 2x2x2 to 3x3x3 indices
        # U rows: i*m+j. For 2x2: i in 0,1, j in 0,1. Same as 3x3.
        # V rows: j*p+k. For 2x2: j in 0,1, k in 0,1. Same as 3x3.
        # W rows: k*n+i. For 2x2: k in 0,1, i in 0,1. Same as 3x3.
        
        blocks.append((U22, V22, W22))
        
        # A21 B12: 2x1x2
        U21, V21, W21 = get_simple_factors(2, 1, 2)
        # Map to 3x3x3
        # U: i in 0,1; j=2. Index i*3+2.
        # V: j=2; k in 0,1. Index 2*3+k.
        # W: k in 0,1; i in 0,1. Index k*3+i.
        
        U_new = np.zeros((9, 2))
        V_new = np.zeros((9, 2))
        W_new = np.zeros((9, 2))
        
        for i in range(2):
            U_new[i * 3 + 2, 0] = U21[i * 1 + 0, 0] # U21 is 2x1
            U_new[i * 3 + 2, 1] = U21[i * 1 + 0, 1]
            
        for k in range(2):
            V_new[2 * 3 + k, 0] = V21[0 * 2 + k, 0] # V21 is 1x2
            V_new[2 * 3 + k, 1] = V21[0 * 2 + k, 1]
            
        for k in range(2):
            for i in range(2):
                W_new[k * 3 + i, 0] = W21[k * 2 + i, 0]
                W_new[k * 3 + i, 1] = W21[k * 2 + i, 1]
                
        blocks.append((U_new, V_new, W_new))
        
        # C21 = A22 B21 + A21 B11
        # A22 B21: 2x2x1
        U22B21, V22B21, W22B21 = get_simple_factors(2, 2, 1)
        # Map to 3x3x3
        # U: i in 0,1; j in 0,1.
        # V: j in 0,1; k=2. Index j*3+2.
        # W: k=2; i in 0,1. Index 2*3+i.
        
        U_new = np.zeros((9, 4))
        V_new = np.zeros((9, 4))
        W_new = np.zeros((9, 4))
        
        for i in range(2):
            for j in range(2):
                U_new[i * 3 + j, 0] = U22B21[i * 2 + j, 0]
                U_new[i * 3 + j, 1] = U22B21[i * 2 + j, 1]
                U_new[i * 3 + j, 2] = U22B21[i * 2 + j, 2]
                U_new[i * 3 + j, 3] = U22B21[i * 2 + j, 3]
                
        for j in range(2):
            V_new[j * 3 + 2, 0] = V22B21[j * 1 + 0, 0]
            V_new[j * 3 + 2, 1] = V22B21[j * 1 + 0, 1]
            V_new[j * 3 + 2, 2] = V22B21[j * 1 + 0, 2]
            V_new[j * 3 + 2, 3] = V22B21[j * 1 + 0, 3]
            
        for i in range(2):
            W_new[2 * 3 + i, 0] = W22B21[0 * 2 + i, 0]
            W_new[2 * 3 + i, 1] = W22B21[0 * 2 + i, 1]
            W_new[2 * 3 + i, 2] = W22B21[0 * 2 + i, 2]
            W_new[2 * 3 + i, 3] = W22B21[0 * 2 + i, 3]
            
        blocks.append((U_new, V_new, W_new))
        
        # A21 B11: 2x1x1
        U21B11, V21B11, W21B11 = get_simple_factors(2, 1, 1)
        # Map
        U_new = np.zeros((9, 1))
        V_new = np.zeros((9, 1))
        W_new = np.zeros((9, 1))
        
        for i in range(2):
            U_new[i * 3 + 2, 0] = U21B11[i * 1 + 0, 0]
        V_new[2 * 3 + 2, 0] = V21B11[0 * 1 + 0, 0]
        for i in range(2):
            W_new[2 * 3 + i, 0] = W21B11[0 * 2 + i, 0]
            
        blocks.append((U_new, V_new, W_new))
        
        # C12 = A12 B22 + A11 B12
        # A12 B22: 1x2x2
        U12B22, V12B22, W12B22 = get_simple_factors(1, 2, 2)
        # Map
        U_new = np.zeros((9, 4))
        V_new = np.zeros((9, 4))
        W_new = np.zeros((9, 4))
        
        for j in range(2):
            U_new[2 * 3 + j, 0] = U12B22[0 * 2 + j, 0]
            U_new[2 * 3 + j, 1] = U12B22[0 * 2 + j, 1]
            U_new[2 * 3 + j, 2] = U12B22[0 * 2 + j, 2]
            U_new[2 * 3 + j, 3] = U12B22[0 * 2 + j, 3]
            
        for j in range(2):
            for k in range(2):
                V_new[j * 3 + k, 0] = V12B22[j * 2 + k, 0]
                V_new[j * 3 + k, 1] = V12B22[j * 2 + k, 1]
                V_new[j * 3 + k, 2] = V12B22[j * 2 + k, 2]
                V_new[j * 3 + k, 3] = V12B22[j * 2 + k, 3]
                
        for k in range(2):
            W_new[k * 3 + 2, 0] = W12B22[k * 1 + 0, 0]
            W_new[k * 3 + 2, 1] = W12B22[k * 1 + 0, 1]
            W_new[k * 3 + 2, 2] = W12B22[k * 1 + 0, 2]
            W_new[k * 3 + 2, 3] = W12B22[k * 1 + 0, 3]
            
        blocks.append((U_new, V_new, W_new))
        
        # A11 B12: 1x1x2
        U11B12, V11B12, W11B12 = get_simple_factors(1, 1, 2)
        # Map
        U_new = np.zeros((9, 2))
        V_new = np.zeros((9, 2))
        W_new = np.zeros((9, 2))
        
        U_new[2 * 3 + 2, 0] = U11B12[0 * 1 + 0, 0]
        U_new[2 * 3 + 2, 1] = U11B12[0 * 1 + 0, 1]
        
        for k in range(2):
            V_new[2 * 3 + k, 0] = V11B12[0 * 2 + k, 0]
            V_new[2 * 3 + k, 1] = V11B12[0 * 2 + k, 1]
            
        for k in range(2):
            W_new[k * 3 + 2, 0] = W11B12[k * 1 + 0, 0]
            W_new[k * 3 + 2, 1] = W11B12[k * 1 + 0, 1]
            
        blocks.append((U_new, V_new, W_new))
        
        # C11 = A12 B21 + A11 B11
        # A12 B21: 1x2x1
        U12B21, V12B21, W12B21 = get_simple_factors(1, 2, 1)
        # Map
        U_new = np.zeros((9, 2))
        V_new = np.zeros((9, 2))
        W_new = np.zeros((9, 2))
        
        for j in range(2):
            U_new[2 * 3 + j, 0] = U12B21[0 * 2 + j, 0]
            U_new[2 * 3 + j, 1] = U12B21[0 * 2 + j, 1]
            
        for j in range(2):
            V_new[j * 3 + 2, 0] = V12B21[j * 1 + 0, 0]
            V_new[j * 3 + 2, 1] = V12B21[j * 1 + 0, 1]
            
        W_new[2 * 3 + 2, 0] = W12B21[0 * 1 + 0, 0]
        W_new[2 * 3 + 2, 1] = W12B21[0 * 1 + 0, 1]
            
        blocks.append((U_new, V_new, W_new))
        
        # A11 B11: 1x1x1
        U11B11, V11B11, W11B11 = get_simple_factors(1, 1, 1)
        # Map
        U_new = np.zeros((9, 1))
        V_new = np.zeros((9, 1))
        W_new = np.zeros((9, 1))
        
        U_new[2 * 3 + 2, 0] = U11B11[0, 0]
        V_new[2 * 3 + 2, 0] = V11B11[0, 0]
        W_new[2 * 3 + 2, 0] = W11B11[0, 0]
            
        blocks.append((U_new, V_new, W_new))
        
        # Concatenate
        U_total = np.hstack([b[0] for b in blocks])
        V_total = np.hstack([b[1] for b in blocks])
        W_total = np.hstack([b[2] for b in blocks])
        
        return U_total, V_total, W_total

    U333, V333, W333 = get_3x3x3_factors()
    rank333 = U333.shape[1]
    
    # Now build 3x3x6
    # Split p=6 into two p=3 parts.
    # U is same for both.
    # V is split into two blocks of 9 rows.
    # W is split into two blocks of 9 rows.
    
    U6 = np.hstack([U333, U333])
    V6 = np.vstack([V333, V333])
    W6 = np.vstack([W333, W333])
    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
