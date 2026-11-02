# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 multiplication with rank 7."""
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i,j], V[j,k], W[k,i]
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][3] = 1 # W[k,i]. C00 (k=0,i=0), C11 (k=1,i=1)
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][2] = 1 # C10 (k=0,i=1)
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][0] = 1 # C01 (k=1,i=0)
    
    # M4
    U[3][3] = 1
    V[0][3] = -1; V[2][3] = 1
    W[0][0] += -1 # C00 -= M4
    W[0][2] += 1  # C10 += M4
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][0] += 1  # C01 += M5
    
    # M6
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[0][0] += 1  # C00 += M6
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][3] += 1  # C11 += M7
    
    return U, V, W

def _rank1_2x1x2():
    """Return U, V, W for 2x1x2 multiplication with rank 1."""
    U = [[0], [0]] # 2x1
    V = [[0], [0]] # 1x2
    W = [[0, 0], [0, 0]] # 2x2
    
    U[0][0] = 1
    U[1][0] = 1
    V[0][0] = 1
    V[0][1] = 1
    W[0][0] = 1
    W[0][1] = 1
    W[1][0] = 1
    W[1][1] = 1
    
    return U, V, W

def construct():
    n, m, p = 2, 3, 6
    rank = 24
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Part 1: j in {0,1}, k in {0,1}, {2,3}, {4,5}
    # Use Strassen 2x2x2
    U_s, V_s, W_s = _strassen_2x2x2()
    for k_offset in [0, 2, 4]:
        for c in range(7):
            # Map local indices to global
            # Local i in {0,1}, j in {0,1}, k in {0,1}
            # Global i = i, j = j, k = k + k_offset
            for i in range(2):
                for j in range(2):
                    U[i*m + j][col] += U_s[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V[j*p + (k + k_offset)][col] += V_s[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W[(k + k_offset)*n + i][col] += W_s[k*2+i][c]
            col += 1
            
    # Part 2: j in {2}, k in {0,1}, {2,3}, {4,5}
    # Use Rank 1 2x1x2
    U_r, V_r, W_r = _rank1_2x1x2()
    for k_offset in [0, 2, 4]:
        for c in range(1):
            # Local i in {0,1}, j=0, k in {0,1}
            # Global i = i, j = 2, k = k + k_offset
            for i in range(2):
                U[i*m + 2][col] += U_r[i][c]
            for k in range(2):
                V[2*p + (k + k_offset)][col] += V_r[0][c]
            for k in range(2):
                for i in range(2):
                    W[(k + k_offset)*n + i][col] += W_r[k][c]
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
