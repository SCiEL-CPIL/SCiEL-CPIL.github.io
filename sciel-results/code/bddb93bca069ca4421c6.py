import numpy as np

def _strassen_base_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are 4x7 matrices.
    U row: i*2+j
    V row: j*2+k
    W row: k*2+i
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices:
    # A: A00 (i=0,j=0), A01 (i=0,j=1), A10 (i=1,j=0), A11 (i=1,j=1)
    # B: B00 (j=0,k=0), B01 (j=0,k=1), B10 (j=1,k=0), B11 (j=1,k=1)
    # C: C00 (k=0,i=0), C01 (k=0,i=1), C10 (k=1,i=0), C11 (k=1,i=1)
    
    # U rows: 00, 01, 10, 11
    # V rows: 00, 01, 10, 11
    # W rows: 00, 01, 10, 11 (k*2+i)
    
    # M1
    # A: A00, A11. B: B00, B11. C: C00(+), C11(+)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    # A: A10, A11. B: B00. C: C10(+), C11(-)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3
    # A: A00. B: B01, B11(-). C: C01(+), C11(+)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4
    # A: A11. B: B10, B00(-). C: C00(+), C10(+)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    # A: A00, A01. B: B11. C: C00(-), C01(+)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6
    # A: A10, A00(-). B: B00, B01. C: C11(+)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    # A: A01, A11(-). B: B10, B11. C: C00(+)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _strassen_decomp(n):
    if n == 2:
        return _strassen_base_2x2x2()
    
    k = n // 2
    U_sub, V_sub, W_sub = _strassen_decomp(k)
    rank_sub = len(U_sub[0])
    rank = 7 * rank_sub
    
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    # Define the 7 steps
    # Each step: (U_blocks, V_blocks, W_blocks)
    # U_blocks: list of (I, J, coeff)
    # V_blocks: list of (J, K, coeff)
    # W_blocks: list of (K, I, coeff)
    
    steps = [
        # M1
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        # M2
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(1,0,1), (1,1,-1)]
        },
        # M3
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(0,1,1), (1,1,1)]
        },
        # M4
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (1,0,1)]
        },
        # M5
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (0,1,1)]
        },
        # M6
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        # M7
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    for s_idx, step in enumerate(steps):
        base_col = s_idx * rank_sub
        # Fill U
        for (I, J, coeff) in step['U']:
            for i in range(k):
                for j in range(k):
                    u_row = I * k + i
                    u_col_local = i * k + j
                    v_row = J * k + j # Not used for U
                    # U_sub row is i*k+j
                    U[u_row * n + (J * k + j)][base_col + u_col_local] += coeff * U_sub[u_col_local] # Wait, U_sub is (k^2, rank_sub)
                    # Correction: U_sub index is (i_sub, j_sub) flattened.
                    # U_sub[i_sub * k + j_sub]
                    
        # Let's redo the filling logic carefully.
        # U_sub has shape (k*k, rank_sub).
        # U has shape (n*n, rank).
        # For a given step s and sub-column c:
        # U[ (I*k+i) * n + (J*k+j) ][ s*rank_sub + c ] += coeff * U_sub[ i*k+j ][c]
        
        for (I, J, coeff) in step['U']:
            for i in range(k):
                for j in range(k):
                    u_row = (I * k + i) * n + (J * k + j)
                    u_sub_row = i * k + j
                    for c in range(rank_sub):
                        U[u_row][base_col + c] += coeff * U_sub[u_sub_row][c]
                        
        for (J, K, coeff) in step['V']:
            for j in range(k):
                for kk in range(k):
                    v_row = (J * k + j) * n + (K * k + kk)
                    v_sub_row = j * k + kk
                    for c in range(rank_sub):
                        V[v_row][base_col + c] += coeff * V_sub[v_sub_row][c]
                        
        for (K, I, coeff) in step['W']:
            for kk in range(k):
                for i in range(k):
                    w_row = (K * k + kk) * n + i
                    w_sub_row = kk * k + i
                    for c in range(rank_sub):
                        W[w_row][base_col + c] += coeff * W_sub[w_sub_row][c]
                        
    return U, V, W

def _schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    U1, V1, W1 = _strassen_decomp(4)
    rank1 = len(U1[0])
    
    # Part 2: 4x4x3 using Schoolbook (Rank 48)
    n2, m2, p2 = 4, 4, 3
    U2, V2, W2 = _schoolbook_decomp(n2, m2, p2)
    rank2 = len(U2[0])
    
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Copy Part 1
    for r in range(rank1):
        for i in range(n*m):
            U[i][r] = U1[i][r]
        for j in range(m*4): # V1 is 16x49, corresponds to B1 (4x4)
            # B1 indices: j in 0..3, k in 0..3.
            # In full V (28x...), row is j*7 + k.
            # In V1, row is j*4 + k.
            j1 = j // 4
            k1 = j % 4
            full_row = j1 * 7 + k1
            V[full_row][r] = V1[j][r]
        for k in range(4*n): # W1 is 16x49, corresponds to C1 (4x4)
            # C1 indices: k in 0..3, i in 0..3.
            # In full W (28x...), row is k*4 + i.
            # In W1, row is k*4 + i.
            W[k][r] = W1[k][r]
            
    # Copy Part 2
    for r in range(rank2):
        for i in range(n*m):
            U[i][r + rank1] = U2[i][r]
        for j in range(m*3): # V2 is 12x48, corresponds to B2 (4x3)
            # B2 indices: j in 0..3, k in 0..2.
            # In full V (28x...), row is j*7 + (4+k).
            j2 = j // 3
            k2 = j % 3
            full_row = j2 * 7 + (4 + k2)
            V[full_row][r + rank1] = V2[j][r]
        for k in range(3*n): # W2 is 12x48, corresponds to C2 (4x3)
            # C2 indices: k in 0..2, i in 0..3.
            # In full W (28x...), row is (4+k)*4 + i.
            k2 = k // 4
            i2 = k % 4
            full_row = (4 + k2) * 4 + i2
            W[full_row][r + rank1] = W2[k][r]
            
    return {"U": U, "V": V, "W": W}
