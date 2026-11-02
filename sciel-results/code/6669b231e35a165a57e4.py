import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    # We will build the decomposition by concatenating blocks.
    # Block 1: A1 (2x2) * B1 (2x5)
    # Block 2: A2 (2x2) * B2 (2x5)
    
    # Helper to add a decomposition of a sub-problem to the global matrices
    # sub_U: local U (n_sub * m_sub, rank_sub)
    # sub_V: local V (m_sub * p_sub, rank_sub)
    # sub_W: local W (p_sub * n_sub, rank_sub)
    # i_offset, j_offset, k_offset: offsets for global indices
    # U global index: (i + i_offset) * m + (j + j_offset)
    # V global index: (j + j_offset) * p + (k + k_offset)
    # W global index: (k + k_offset) * n + (i + i_offset)
    
    rank_total = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Strassen 2x2x2 decomposition
    # Local indices: i in 0,1; j in 0,1; k in 0,1
    # U (4, 7), V (4, 7), W (4, 7)
    def get_strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = 1; W[2, 0] = 1 # C00+=M1, C11+=M1, C01-=M1? No.
        # Let's use the standard formula mapping.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: A00, A11 -> B00, B11 -> C00, C11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00, C11. Note W index is k*n+i.
        # C00 is i=0,k=0 -> idx 0. C11 is i=1,k=1 -> idx 1*2+1=3.
        
        # M2: (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10 (i=1,k=0) -> idx 0*2+1=2.
        # C11 -= M2 -> W[3, 1] = -1
        
        # M3: A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01 (i=0,k=1) -> idx 1*2+0=1.
        # C11 += M3 -> W[3, 2] = 1
        
        # M4: A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1 # C00 += M4
        W[2, 3] = 1 # C10 += M4
        
        # M5: (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1 # C00 -= M5
        W[1, 4] = 1 # C01 += M5
        
        # M6: (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C11 += M6
        
        # M7: (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C00 += M7
        W[3, 6] = -1 # C11 -= M7 (Wait, C11 = M1 - M2 + M3 + M6. M7 is not in C11?)
        # Let's re-verify Strassen formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # Yes, M7 is only in C00.
        
        return U, V, W

    # Schoolbook 2x2x1 decomposition
    # Local indices: i in 0,1; j in 0,1; k in 0
    # U (4, 4), V (2, 4), W (2, 4)
    def get_schoolbook_221():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # P1 = A00 B00
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1 # C00 (i=0,k=0) idx 0
        
        # P2 = A01 B10
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1 # C00 += P2
        
        # P3 = A10 B00
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1 # C10 (i=1,k=0) idx 1
        
        # P4 = A11 B10
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1 # C10 += P4
        
        return U, V, W

    # Block 1: A1 (rows 0-1 of A, cols 0-1) * B1 (rows 0-1 of B, cols 0-4)
    # Split B1 into B1a (cols 0), B1b (cols 1-2), B1c (cols 3-4)
    # B1a is 2x1. B1b is 2x2. B1c is 2x2.
    
    # Sub-block 1a: 2x2x1
    U1a, V1a, W1a = get_schoolbook_221()
    # Map to global
    # i_offset=0, j_offset=0, k_offset=0
    # U global rows: 0,1,2,3. V global rows: 0,1. W global rows: 0,1.
    
    # Sub-block 1b: 2x2x2 (cols 1,2)
    U1b, V1b, W1b = get_strassen_222()
    # i_offset=0, j_offset=0, k_offset=1
    
    # Sub-block 1c: 2x2x2 (cols 3,4)
    U1c, V1c, W1c = get_strassen_222()
    # i_offset=0, j_offset=0, k_offset=3
    
    # Block 2: A2 (rows 0-1 of A, cols 2-3) * B2 (rows 2-3 of B, cols 0-4)
    # Split B2 into B2a (cols 0), B2b (cols 1-2), B2c (cols 3-4)
    
    # Sub-block 2a: 2x2x1
    U2a, V2a, W2a = get_schoolbook_221()
    # i_offset=0, j_offset=2, k_offset=0
    
    # Sub-block 2b: 2x2x2 (cols 1,2)
    U2b, V2b, W2b = get_strassen_222()
    # i_offset=0, j_offset=2, k_offset=1
    
    # Sub-block 2c: 2x2x2 (cols 3,4)
    U2c, V2c, W2c = get_strassen_222()
    # i_offset=0, j_offset=2, k_offset=3

    # Total rank = 4 + 7 + 7 + 4 + 7 + 7 = 36
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Add 1a
    U[0:4, col:col+4] = U1a
    V[0:2, col:col+4] = V1a
    W[0:2, col:col+4] = W1a
    col += 4
    
    # Add 1b
    U[0:4, col:col+7] = U1b
    # V rows for j=0,1 and k=1,2.
    # Local V is 2x4 (j in 0,1; k in 0,1).
    # Global V index: j*p + k.
    # j=0, k=1 -> 1. j=0, k=2 -> 2.
    # j=1, k=1 -> 6. j=1, k=2 -> 7.
    V[1, col:col+7] += V1b[0, :]
    V[2, col:col+7] += V1b[0, :]
    V[6, col:col+7] += V1b[1, :]
    V[7, col:col+7] += V1b[1, :]
    # W rows for k=1,2 and i=0,1.
    # Local W is 2x4 (k in 0,1; i in 0,1).
    # Global W index: k*n + i.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    # k=2, i=0 -> 4. k=2, i=1 -> 5.
    W[2, col:col+7] += W1b[0, :]
    W[3, col:col+7] += W1b[0, :]
    W[4, col:col+7] += W1b[1, :]
    W[5, col:col+7] += W1b[1, :]
    col += 7
    
    # Add 1c
    U[0:4, col:col+7] = U1c
    # V rows for j=0,1 and k=3,4.
    V[3, col:col+7] += V1c[0, :]
    V[4, col:col+7] += V1c[0, :]
    V[8, col:col+7] += V1c[1, :]
    V[9, col:col+7] += V1c[1, :]
    # W rows for k=3,4 and i=0,1.
    W[6, col:col+7] += W1c[0, :]
    W[7, col:col+7] += W1c[0, :]
    W[8, col:col+7] += W1c[1, :]
    W[9, col:col+7] += W1c[1, :]
    col += 7
    
    # Add 2a
    # i_offset=0, j_offset=2, k_offset=0
    # U rows: i in 0,1; j in 2,3.
    # Global U index: i*m + j.
    # i=0, j=2 -> 2. i=0, j=3 -> 3.
    # i=1, j=2 -> 6. i=1, j=3 -> 7.
    U[2, col:col+4] = U2a[0, :]
    U[3, col:col+4] = U2a[1, :]
    U[6, col:col+4] = U2a[2, :]
    U[7, col:col+4] = U2a[3, :]
    # V rows: j in 2,3; k in 0.
    # Global V index: j*p + k.
    # j=2, k=0 -> 10. j=3, k=0 -> 15.
    V[10, col:col+4] = V2a[0, :]
    V[15, col:col+4] = V2a[1, :]
    # W rows: k in 0; i in 0,1.
    W[0, col:col+4] += W2a[0, :]
    W[1, col:col+4] += W2a[1, :]
    col += 4
    
    # Add 2b
    # i_offset=0, j_offset=2, k_offset=1
    U[2, col:col+7] = U2b[0, :]
    U[3, col:col+7] = U2b[1, :]
    U[6, col:col+7] = U2b[2, :]
    U[7, col:col+7] = U2b[3, :]
    # V rows: j in 2,3; k in 1,2.
    V[11, col:col+7] += V2b[0, :]
    V[12, col:col+7] += V2b[0, :]
    V[16, col:col+7] += V2b[1, :]
    V[17, col:col+7] += V2b[1, :]
    # W rows: k in 1,2; i in 0,1.
    W[2, col:col+7] += W2b[0, :]
    W[3, col:col+7] += W2b[0, :]
    W[4, col:col+7] += W2b[1, :]
    W[5, col:col+7] += W2b[1, :]
    col += 7
    
    # Add 2c
    # i_offset=0, j_offset=2, k_offset=3
    U[2, col:col+7] = U2c[0, :]
    U[3, col:col+7] = U2c[1, :]
    U[6, col:col+7] = U2c[2, :]
    U[7, col:col+7] = U2c[3, :]
    # V rows: j in 2,3; k in 3,4.
    V[13, col:col+7] += V2c[0, :]
    V[14, col:col+7] += V2c[0, :]
    V[18, col:col+7] += V2c[1, :]
    V[19, col:col+7] += V2c[1, :]
    # W rows: k in 3,4; i in 0,1.
    W[6, col:col+7] += W2c[0, :]
    W[7, col:col+7] += W2c[0, :]
    W[8, col:col+7] += W2c[1, :]
    W[9, col:col+7] += W2c[1, :]
    col += 7

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

