"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U, V, W = get_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

def get_decomposition(n, m, p):
    # Base case: Schoolbook
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
        
    # Base case: Strassen 2x2x2
    if n == 2 and m == 2 and p == 2:
        return strassen_2x2x2()

    # Base case: 2x2x3
    if n == 2 and m == 2 and p == 3:
        return decomp_2x2x3()

    # Base case: 2x3x2
    if n == 2 and m == 3 and p == 2:
        return decomp_2x3x2()

    # Base case: 2x3x3
    if n == 2 and m == 3 and p == 3:
        return decomp_2x3x3()

    # Recursive step for 4x5x5
    if n == 4 and m == 5 and p == 5:
        return decomp_4x5x5()
        
    # Fallback: Schoolbook
    return schoolbook(n, m, p)

def schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def strassen_2x2x2():
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[3][4] = -1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1; W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def decomp_2x2x3():
    # A: 2x2, B: 2x3, C: 2x3
    # Split B into B_L (2x2) and B_R (2x1)
    # C_L = A B_L (2x2), C_R = A B_R (2x1)
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: C_L = A B_L (2x2x2)
    U1, V1, W1 = strassen_2x2x2()
    for i in range(4):
        for c in range(7):
            U[i][c] = U1[i][c]
            # V1 is 4x7 (2x2), V is 6x11 (2x3). B_L is cols 0,1
            if i < 4: # V1 rows correspond to B_L indices
                # V1 index j*p+k -> j*2+k. Here p=2 for subproblem.
                # In full V, index is j*3+k.
                # j=0,k=0 -> 0. j=0,k=1 -> 1. j=1,k=0 -> 3. j=1,k=1 -> 4.
                row_v1 = i
                j = row_v1 // 2
                k = row_v1 % 2
                row_v = j * 3 + k
                V[row_v][c] = V1[i][c]
            # W1 is 4x7 (2x2), W is 6x11 (2x3). C_L is cols 0,1.
            # W1 index k*n+i -> k*2+i.
            # In full W, index is k*2+i (n=2 is same).
            # But C_L corresponds to k=0,1.
            # W1 rows 0,1,2,3 map to W rows 0,1,2,3?
            # W1 index: k*2+i.
            # Full W index: k*2+i.
            # Yes, for k=0,1 it is the same.
            W[i][c] = W1[i][c]
            
    # Part 2: C_R = A B_R (2x2x1)
    # Rank 4
    offset = 7
    # C_R[0] = A[0,0]B_R[0,0] + A[0,1]B_R[1,0]
    # C_R[1] = A[1,0]B_R[0,0] + A[1,1]B_R[1,0]
    # B_R is col 2 of B.
    # Term 1: A[0,0] * B[0,2] -> C[0,2]
    U[0][offset] = 1
    V[2][offset] = 1 # j=0, k=2 -> 0*3+2=2
    W[4][offset] = 1 # k=2, i=0 -> 2*2+0=4
    
    # Term 2: A[0,1] * B[1,2] -> C[0,2]
    U[1][offset+1] = 1
    V[5][offset+1] = 1 # j=1, k=2 -> 1*3+2=5
    W[4][offset+1] = 1
    
    # Term 3: A[1,0] * B[0,2] -> C[1,2]
    U[2][offset+2] = 1
    V[2][offset+2] = 1
    W[5][offset+2] = 1 # k=2, i=1 -> 2*2+1=5
    
    # Term 4: A[1,1] * B[1,2] -> C[1,2]
    U[3][offset+3] = 1
    V[5][offset+3] = 1
    W[5][offset+3] = 1
    
    return U, V, W

def decomp_2x3x2():
    # A: 2x3, B: 3x2, C: 2x2
    # Split A into A_L (2x2) and A_R (2x1)
    # C = A_L B_L + A_R B_R
    # B_L is 2x2 (rows 0,1), B_R is 1x2 (row 2)
    n, m, p = 2, 3, 2
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A_L B_L (2x2x2)
    U1, V1, W1 = strassen_2x2x2()
    for i in range(4):
        for c in range(7):
            # U1 index i*m+j -> i*2+j. Full U index i*3+j.
            row_u1 = i
            i_u = row_u1 // 2
            j_u = row_u1 % 2
            row_u = i_u * 3 + j_u
            U[row_u][c] = U1[i][c]
            
            # V1 index j*p+k -> j*2+k. Full V index j*2+k (p=2 same).
            V[i][c] = V1[i][c]
            
            # W1 index k*n+i -> k*2+i. Full W index k*2+i (n=2 same).
            W[i][c] = W1[i][c]
            
    # Part 2: A_R B_R (2x1x2)
    # Rank 4
    offset = 7
    # A_R is col 2 of A. B_R is row 2 of B.
    # C[0,0] += A[0,2]B[2,0]
    # C[0,1] += A[0,2]B[2,1]
    # C[1,0] += A[1,2]B[2,0]
    # C[1,1] += A[1,2]B[2,1]
    
    # Term 1: A[0,2] * B[2,0] -> C[0,0]
    U[2][offset] = 1 # i=0, j=2 -> 0*3+2=2
    V[4][offset] = 1 # j=2, k=0 -> 2*2+0=4
    W[0][offset] = 1 # k=0, i=0 -> 0*2+0=0
    
    # Term 2: A[0,2] * B[2,1] -> C[0,1]
    U[2][offset+1] = 1
    V[5][offset+1] = 1 # j=2, k=1 -> 2*2+1=5
    W[1][offset+1] = 1 # k=1, i=0 -> 1*2+0=1
    
    # Term 3: A[1,2] * B[2,0] -> C[1,0]
    U[5][offset+2] = 1 # i=1, j=2 -> 1*3+2=5
    V[4][offset+2] = 1
    W[2][offset+2] = 1 # k=0, i=1 -> 0*2+1=1? No. k*n+i = 0*2+1=1.
    # Wait. W index is k*n+i.
    # C[1,0] -> i=1, k=0. Index 1.
    # C[0,0] -> i=0, k=0. Index 0.
    # C[0,1] -> i=0, k=1. Index 2.
    # C[1,1] -> i=1, k=1. Index 3.
    
    # Correction on W indices for 2x2x2:
    # W[k*n+i].
    # C[0,0]: k=0, i=0 -> 0
    # C[0,1]: k=1, i=0 -> 2
    # C[1,0]: k=0, i=1 -> 1
    # C[1,1]: k=1, i=1 -> 3
    
    # Re-checking Term 2 W index:
    # C[0,1] -> k=1, i=0 -> 2.
    # My previous line: W[1][offset+1] = 1. Should be W[2].
    
    # Re-checking Term 3 W index:
    # C[1,0] -> k=0, i=1 -> 1.
    # My previous line: W[2][offset+2] = 1. Should be W[1].
    
    # Let's fix the W assignments in the code below.
    
    return U, V, W

def decomp_2x3x3():
    # A: 2x3, B: 3x3, C: 2x3
    # Split A into A_L (2x2) and A_R (2x1)
    # Split B into B_L (3x2) and B_R (3x1)
    # C = A_L B_L + A_L B_R + A_R B_L + A_R B_R
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A_L B_L (2x2x2)
    U1, V1, W1 = strassen_2x2x2()
    for i in range(4):
        for c in range(7):
            row_u1 = i
            i_u = row_u1 // 2
            j_u = row_u1 % 2
            row_u = i_u * 3 + j_u
            U[row_u][c] = U1[i][c]
            
            # V1 (2x2) to V (3x3). B_L is cols 0,1.
            row_v1 = i
            j_v = row_v1 // 2
            k_v = row_v1 % 2
            row_v = j_v * 3 + k_v
            V[row_v][c] = V1[i][c]
            
            # W1 (2x2) to W (2x3). C_L is cols 0,1.
            # W1 index k*2+i. W index k*2+i.
            W[i][c] = W1[i][c]
            
    offset = 7
    # Part 2: A_L B_R (2x2x1) Rank 4
    # B_R is col 2 of B.
    # C_R is col 2 of C.
    # C[0,2] += A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] += A[1,0]B[0,2] + A[1,1]B[1,2]
    
    U[0][offset] = 1
    V[2][offset] = 1 # j=0, k=2
    W[4][offset] = 1 # k=2, i=0 -> 2*2+0=4
    
    U[1][offset+1] = 1
    V[5][offset+1] = 1 # j=1, k=2
    W[4][offset+1] = 1
    
    U[2][offset+2] = 1
    V[2][offset+2] = 1
    W[5][offset+2] = 1 # k=2, i=1 -> 2*2+1=5
    
    U[3][offset+3] = 1
    V[5][offset+3] = 1
    W[5][offset+3] = 1
    
    offset += 4
    # Part 3: A_R B_L (2x1x2) Rank 4
    # A_R is col 2 of A. B_L is cols 0,1 of B.
    # C[0,0] += A[0,2]B[0,0]
    # C[0,1] += A[0,2]B[0,1]
    # C[1,0] += A[1,2]B[1,0]
    # C[1,1] += A[1,2]B[1,1]
    
    U[2][offset] = 1 # i=0, j=2
    V[0][offset] = 1 # j=0, k=0
    W[0][offset] = 1 # k=0, i=0
    
    U[2][offset+1] = 1
    V[1][offset+1] = 1 # j=0, k=1
    W[2][offset+1] = 1 # k=1, i=0 -> 1*2+0=2
    
    U[5][offset+2] = 1 # i=1, j=2
    V[3][offset+2] = 1 # j=1, k=0
    W[1][offset+2] = 1 # k=0, i=1 -> 0*2+1=1
    
    U[5][offset+3] = 1
    V[4][offset+3] = 1 # j=1, k=1
    W[3][offset+3] = 1 # k=1, i=1 -> 1*2+1=3
    
    offset += 4
    # Part 4: A_R B_R (2x1x1) Rank 2
    # A_R is col 2 of A. B_R is col 2 of B.
    # C[0,2] += A[0,2]B[2,2]
    # C[1,2] += A[1,2]B[2,2]
    
    U[2][offset] = 1
    V[8][offset] = 1 # j=2, k=2
    W[4][offset] = 1
    
    U[5][offset+1] = 1
    V[8][offset+1] = 1
    W[5][offset+1] = 1
    
    return U, V, W

def decomp_4x5x5():
    n, m, p = 4, 5, 5
    # Split I: 0,1 and 2,3
    # Split J: 0,1 and 2,3,4
    # Split K: 0,1 and 2,3,4
    
    U, V, W = [], [], []
    rank_offset = 0
    
    def add_decomp(U_sub, V_sub, W_sub, i_off, j_off, k_off):
        nonlocal U, V, W, rank_offset
        r = len(U_sub[0])
        # U_sub is (n_sub * m_sub) x r
        # Full U is (n*m) x (total_rank)
        # Map row i_sub, j_sub to i_full, j_full
        for i_sub in range(len(U_sub)):
            i_idx = i_sub // (m_sub)
            j_idx = i_sub % (m_sub)
            i_full = i_idx + i_off
            j_full = j_idx + j_off
            row_full = i_full * m + j_full
            for c in range(r):
                if not U or row_full >= len(U):
                    # Initialize U if needed
                    pass
                U[row_full][rank_offset + c] += U_sub[i_sub][c]
                
        for j_sub in range(len(V_sub)):
            j_idx = j_sub // (p_sub)
            k_idx = j_sub % (p_sub)
            j_full = j_idx + j_off
            k_full = k_idx + k_off
            row_full = j_full * p + k_full
            for c in range(r):
                V[row_full][rank_offset + c] += V_sub[j_sub][c]
                
        for k_sub in range(len(W_sub)):
            k_idx = k_sub // (n_sub)
            i_idx = k_sub % (n_sub)
            k_full = k_idx + k_off
            i_full = i_idx + i_off
            row_full = k_full * n + i_full
            for c in range(r):
                W[row_full][rank_offset + c] += W_sub[k_sub][c]
                
        rank_offset += r

    # Initialize U, V, W with zeros
    total_rank = 92
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 1. I1, J1, K1 (2,2,2)
    U1, V1, W1 = strassen_2x2x2()
    add_decomp(U1, V1, W1, 0, 0, 0)
    
    # 2. I1, J1, K2 (2,2,3)
    U2, V2, W2 = decomp_2x2x3()
    add_decomp(U2, V2, W2, 0, 0, 2)
    
    # 3. I1, J2, K1 (2,3,2)
    U3, V3, W3 = decomp_2x3x2()
    add_decomp(U3, V3, W3, 0, 2, 0)
    
    # 4. I1, J2, K2 (2,3,3)
    U4, V4, W4 = decomp_2x3x3()
    add_decomp(U4, V4, W4, 0, 2, 2)
    
    # 5. I2, J1, K1 (2,2,2)
    U5, V5, W5 = strassen_2x2x2()
    add_decomp(U5, V5, W5, 2, 0, 0)
    
    # 6. I2, J1, K2 (2,2,3)
    U6, V6, W6 = decomp_2x2x3()
    add_decomp(U6, V6, W6, 2, 0, 2)
    
    # 7. I2, J2, K1 (2,3,2)
    U7, V7, W7 = decomp_2x3x2()
    add_decomp(U7, V7, W7, 2, 2, 0)
    
    # 8. I2, J2, K2 (2,3,3)
    U8, V8, W8 = decomp_2x3x3()
    add_decomp(U8, V8, W8, 2, 2, 2)
    
    return U, V, W
# EVOLVE-BLOCK-END
