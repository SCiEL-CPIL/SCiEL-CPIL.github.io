# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 3, 4
    # Rank 22 solution using blocking
    # A is 2x3, B is 3x4
    # A = [A1 (2x2), A2 (2x1)]
    # B = [B1 (2x4); B2 (1x4)]
    # C = A1*B1 + A2*B2
    
    # We need to construct U, V, W for the whole system.
    # U rows: (i,j) for i in 0..1, j in 0..2 -> 6 rows
    # V rows: (j,k) for j in 0..2, k in 0..3 -> 12 rows
    # W rows: (k,i) for k in 0..3, i in 0..1 -> 8 rows
    
    # We will use 14 columns for A1*B1 and 8 columns for A2*B2.
    # Total rank = 22.
    
    rank = 22
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(8)]
    
    # Helper to add a Strassen decomposition for 2x2 * 2x2
    # Returns U_sub, V_sub, W_sub for the sub-problem
    # U_sub shape (4, 7), V_sub shape (4, 7), W_sub shape (4, 7)
    def strassen_2x2():
        # Strassen coefficients for 2x2 matrix multiplication
        # C11 = M1 + M4 - M5 + M6
        # C12 = M2 + M5
        # C21 = M3 + M4
        # C22 = M1 - M2 + M3 + M7
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # Indices for U, V, W are 0,1,2,3 corresponding to (0,0),(0,1),(1,0),(1,1)
        # U rows: A entries
        # V rows: B entries
        # W rows: C entries
        
        # M1
        u[0][0] = 1; u[3][0] = 1 # A00, A11
        v[0][0] = 1; v[3][0] = 1 # B00, B11
        w[0][0] = 1; w[3][0] = 1 # C00, C11
        
        # M2
        u[2][1] = 1; u[3][1] = 1 # A10, A11
        v[0][1] = 1 # B00
        w[1][1] = 1 # C01
        
        # M3
        u[0][2] = 1 # A00
        v[2][2] = 1; v[3][2] = -1 # B10, B11
        w[2][2] = 1 # C10
        
        # M4
        u[3][3] = 1 # A11
        v[1][3] = 1; v[0][3] = -1 # B01, B00
        w[0][3] = 1; w[2][3] = 1 # C00, C10
        
        # M5
        u[0][4] = 1; u[1][4] = 1 # A00, A01
        v[3][4] = 1 # B11
        w[1][4] = 1 # C01
        
        # M6
        u[2][5] = 1; u[0][5] = -1 # A10, A00
        v[0][5] = 1; v[1][5] = 1 # B00, B01
        w[3][5] = 1 # C11
        
        # M7
        u[1][6] = 1; u[3][6] = -1 # A01, A11
        v[2][6] = 1; v[3][6] = 1 # B10, B11
        w[3][6] = 1 # C11
        
        return u, v, w

    u1, v1, w1 = strassen_2x2()
    u2, v2, w2 = strassen_2x2()
    
    # Map A1 (2x2) to U rows 0..3 (j=0,1)
    # Map B1 (2x4) to V rows 0..7 (j=0,1; k=0..3)
    # Split B1 into B1L (k=0,1) and B1R (k=2,3)
    
    # Add Strassen 1 (for B1L)
    for r in range(7):
        for i in range(4):
            U[i][r] += u1[i][r]
        for j in range(4):
            # V row j in sub-problem corresponds to (j//2, j%2) in B1L?
            # Sub-problem V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # B1L indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global V indices: (j,k) -> j*p+k
            # (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
            v_idx = [0, 1, 4, 5][j]
            V[v_idx][r] += v1[j][r]
        for k in range(4):
            # W row k in sub-problem corresponds to (k//2, k%2) in C1L?
            # Sub-problem W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # C1L indices: (0,0), (0,1), (1,0), (1,1)
            # Global W indices: (k,i) -> k*n+i
            # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            w_idx = [0, 1, 2, 3][k]
            W[w_idx][r] += w1[k][r]
            
    # Add Strassen 2 (for B1R)
    for r in range(7):
        col = 7 + r
        for i in range(4):
            U[i][col] += u2[i][r]
        for j in range(4):
            # B1R indices: 0:(0,2), 1:(0,3), 2:(1,2), 3:(1,3)
            # Global V indices: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
            v_idx = [2, 3, 6, 7][j]
            V[v_idx][col] += v2[j][r]
        for k in range(4):
            # C1R indices: (0,2), (0,3), (1,2), (1,3)
            # Global W indices: (2,0)->4, (2,1)->5, (3,0)->6, (3,1)->7
            w_idx = [4, 5, 6, 7][k]
            W[w_idx][col] += w2[k][r]

    # Add Schoolbook for A2 (2x1) * B2 (1x4)
    # A2 rows: 4:(0,2), 5:(1,2)
    # B2 rows: 8:(2,0), 9:(2,1), 10:(2,2), 11:(2,3)
    # C2 entries: (0,0)..(1,3)
    for i in range(2):
        for k in range(4):
            col = 14 + i * 4 + k
            U[4 + i][col] = 1 # A[i, 2]
            V[8 + k][col] = 1 # B[2, k]
            W[k * 2 + i][col] = 1 # C[i, k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
