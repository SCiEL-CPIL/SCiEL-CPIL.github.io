import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are 4x7 matrices.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (a+d)(e+h)
    # a=A00, d=A11, e=B00, h=B11
    # C00 += M1, C11 += M1
    U[0, 0] = 1 # A00
    U[3, 0] = 1 # A11
    V[0, 0] = 1 # B00
    V[3, 0] = 1 # B11
    W[0, 0] = 1 # C00 (k=0, i=0)
    W[3, 0] = 1 # C11 (k=1, i=1)
    
    # M2 = (c+d)e
    # c=A10, d=A11, e=B00
    # C10 += M2
    U[2, 1] = 1 # A10
    U[3, 1] = 1 # A11
    V[0, 1] = 1 # B00
    W[2, 1] = 1 # C10 (k=0, i=1)
    
    # M3 = a(f-h)
    # a=A00, f=B01, h=B11
    # C01 += M3
    U[0, 2] = 1 # A00
    V[1, 2] = 1 # B01
    V[3, 2] = -1 # B11
    W[1, 2] = 1 # C01 (k=1, i=0)
    
    # M4 = d(g-f)
    # d=A11, g=B10, f=B01
    # C00 += M4, C10 += M4
    U[3, 3] = 1 # A11
    V[2, 3] = 1 # B10
    V[1, 3] = -1 # B01
    W[0, 3] = 1 # C00
    W[2, 3] = 1 # C10
    
    # M5 = (a+b)h
    # a=A00, b=A01, h=B11
    # C00 -= M5, C01 += M5
    U[0, 4] = 1 # A00
    U[1, 4] = 1 # A01
    V[3, 4] = 1 # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1 # C01
    
    # M6 = (c-a)(e+f)
    # c=A10, a=A00, e=B00, f=B01
    # C11 += M6
    U[2, 5] = 1 # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1 # B00
    V[1, 5] = 1 # B01
    W[3, 5] = 1 # C11
    
    # M7 = (b-d)(g+h)
    # b=A01, d=A11, g=B10, h=B11
    # C00 += M7
    U[1, 6] = 1 # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1 # B10
    V[3, 6] = 1 # B11
    W[0, 6] = 1 # C00
    
    return U, V, W

def _schoolbook_factors(n, m, p):
    """
    Returns U, V, W for schoolbook decomposition of n x m x p.
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    
    # We decompose C = A B where A is 4x5, B is 5x5.
    # Split A = [A1 (4x4), A2 (4x1)]
    # Split B = [[B11 (4x4), B12 (4x1)], [B21 (1x4), B22 (1x1)]]
    
    # C = A1 B11 + A1 B12 + A2 B21 + A2 B22
    
    # 1. A1 B11: 4x4 * 4x4. Rank 49 (Strassen)
    U2, V2, W2 = _strassen_2x2_factors()
    # Kronecker product to get 4x4x4 factors
    # U4 = U2 (x) U2
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    # Check sizes: U2 is 4x7. U4 is 16x49.
    # U4 corresponds to A1 (4x4). Rows are i*4+j.
    # V4 corresponds to B11 (4x4). Rows are j*4+k.
    # W4 corresponds to C1 (4x4). Rows are k*4+i.
    
    # 2. A1 B12: 4x4 * 4x1. Rank 16 (Schoolbook)
    # A1 is 4x4, B12 is 4x1.
    # This is multiplying a 4x4 matrix by a 4x1 vector.
    # We can use schoolbook for n=4, m=4, p=1.
    U_a1b12, V_a1b12, W_a1b12 = _schoolbook_factors(4, 4, 1)
    # U_a1b12 is 16x16. Rows i*4+j.
    # V_a1b12 is 4x16. Rows j*1+k (k=0).
    # W_a1b12 is 4x16. Rows k*4+i (k=0).
    
    # 3. A2 B21: 4x1 * 1x4. Rank 16 (Schoolbook)
    # A2 is 4x1, B21 is 1x4.
    # n=4, m=1, p=4.
    U_a2b21, V_a2b21, W_a2b21 = _schoolbook_factors(4, 1, 4)
    # U_a2b21 is 4x16. Rows i*1+j (j=0).
    # V_a2b21 is 4x16. Rows j*4+k (j=0).
    # W_a2b21 is 16x16. Rows k*4+i.
    
    # 4. A2 B22: 4x1 * 1x1. Rank 4 (Schoolbook)
    # n=4, m=1, p=1.
    U_a2b22, V_a2b22, W_a2b22 = _schoolbook_factors(4, 1, 1)
    # U_a2b22 is 4x4.
    # V_a2b22 is 1x4.
    # W_a2b22 is 4x4.
    
    # Now we need to assemble the full U, V, W for 4x5x5.
    # Full U size: 4*5 = 20 rows. 49+16+16+4 = 85 columns.
    # Full V size: 5*5 = 25 rows. 49+16+16+4 = 85 columns.
    # Full W size: 5*4 = 20 rows. 49+16+16+4 = 85 columns.
    
    total_rank = 49 + 16 + 16 + 4
    U = np.zeros((20, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((20, total_rank))
    
    # Part 1: A1 B11 (cols 0-48)
    # A1 occupies rows 0-15 of U (i*5+j for j=0..3? No.
    # A is 4x5. A1 is columns 0-3. A2 is column 4.
    # U rows: i*5 + j.
    # For A1 (j=0..3), rows are i*5 + j.
    # U4 rows are i*4 + j.
    # We need to map U4 rows to U rows.
    # U4 row i*4+j corresponds to A1[i,j].
    # In full U, A1[i,j] is at row i*5 + j.
    
    for i in range(4):
        for j in range(4):
            U[i*5 + j, 0:49] = U4[i*4 + j, :]
    
    # V4 rows: j*4 + k.
    # In full V, B11[j,k] is at row j*5 + k.
    for j in range(4):
        for k in range(4):
            V[j*5 + k, 0:49] = V4[j*4 + k, :]
            
    # W4 rows: k*4 + i.
    # In full W, C1[k,i] is at row k*4 + i?
    # W index is k*n + i. Here n=4 for the sub-problem?
    # No, the full W index is k*4 + i (since n=4).
    # So W4 rows map directly to W rows 0-15.
    W[0:16, 0:49] = W4
    
    # Part 2: A1 B12 (cols 49-64)
    # A1 is 4x4. B12 is 4x1 (column 4 of B).
    # U_a1b12 rows: i*4 + j.
    # Map to U rows i*5 + j.
    for i in range(4):
        for j in range(4):
            U[i*5 + j, 49:65] = U_a1b12[i*4 + j, :]
            
    # V_a1b12 rows: j*1 + 0 = j.
    # In full V, B12[j, 0] is at row j*5 + 4.
    for j in range(4):
        V[j*5 + 4, 49:65] = V_a1b12[j, :]
        
    # W_a1b12 rows: 0*4 + i = i.
    # In full W, C_part2[k=0, i] is at row 0*4 + i = i.
    W[0:4, 49:65] = W_a1b12
    
    # Part 3: A2 B21 (cols 65-80)
    # A2 is 4x1 (column 4 of A). B21 is 1x4 (row 4 of B).
    # U_a2b21 rows: i*1 + 0 = i.
    # In full U, A2[i, 0] is at row i*5 + 4.
    for i in range(4):
        U[i*5 + 4, 65:81] = U_a2b21[i, :]
        
    # V_a2b21 rows: 0*4 + k = k.
    # In full V, B21[0, k] is at row 4*5 + k = 20 + k.
    for k in range(4):
        V[20 + k, 65:81] = V_a2b21[k, :]
        
    # W_a2b21 rows: k*4 + i.
    # In full W, C_part3[k, i] is at row k*4 + i.
    W[0:16, 65:81] = W_a2b21
    
    # Part 4: A2 B22 (cols 81-84)
    # A2 is 4x1. B22 is 1x1.
    # U_a2b22 rows: i.
    # In full U, A2[i, 0] is at row i*5 + 4.
    for i in range(4):
        U[i*5 + 4, 81:85] = U_a2b22[i, :]
        
    # V_a2b22 rows: 0.
    # In full V, B22[0, 0] is at row 4*5 + 4 = 24.
    V[24, 81:85] = V_a2b22[0, :]
        
    # W_a2b22 rows: k=0, i. Row i.
    # In full W, C_part4[0, i] is at row i.
    W[0:4, 81:85] = W_a2b22
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
