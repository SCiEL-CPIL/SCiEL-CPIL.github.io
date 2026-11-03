import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (rank 7).
    U, V, W are lists of lists (integers).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products
    # p1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1 # C00, C11
    
    # p2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1 # C10
    
    # p3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1 # C01
    
    # p4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1 # C00, C10
    
    # p5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1 # C00, C01
    
    # p6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # C11
    
    # p7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1 # C00, C01
    
    return U, V, W

def _kron_with_permutation(U1, U2, perm_func):
    """
    Computes Kronecker product U1 x U2 and applies row permutation.
    """
    n1, r1 = len(U1), len(U1[0])
    n2, r2 = len(U2), len(U2[0])
    n = n1 * n2
    r = r1 * r2
    U = [[0] * r for _ in range(n)]
    
    for i1 in range(n1):
        for j1 in range(r1):
            if U1[i1][j1] != 0:
                for i2 in range(n2):
                    for j2 in range(r2):
                        if U2[i2][j2] != 0:
                            # Kronecker index
                            kron_i = i1 * n2 + i2
                            kron_j = j1 * r2 + j2
                            # Permute
                            new_i = perm_func(i1, i2)
                            U[new_i][kron_j] += U1[i1][j1] * U2[i2][j2]
    return U

def _perm_u(i1, i2):
    # i1 is (i_1, j_1) flattened? No, i1 is row index of U1.
    # U1 row index = i_1 * 2 + j_1
    # U2 row index = i_2 * 2 + j_2
    # We need to map (i_1, j_1, i_2, j_2) to target index.
    # But here i1 is the row index of U1, which is i_1 * 2 + j_1.
    # So we need to unpack i1, i2.
    i_1 = i1 // 2
    j_1 = i1 % 2
    i_2 = i2 // 2
    j_2 = i2 % 2
    # Target index: (2*i_1 + i_2)*4 + (2*j_1 + j_2)
    return (2 * i_1 + i_2) * 4 + (2 * j_1 + j_2)

def _perm_v(j1, j2):
    # V1 row index = j_1 * 2 + k_1
    # V2 row index = j_2 * 2 + k_2
    j_1 = j1 // 2
    k_1 = j1 % 2
    j_2 = j2 // 2
    k_2 = j2 % 2
    # Target index: (2*j_1 + j_2)*4 + (2*k_1 + k_2)
    return (2 * j_1 + j_2) * 4 + (2 * k_1 + k_2)

def _perm_w(k1, k2):
    # W1 row index = k_1 * 2 + i_1
    # W2 row index = k_2 * 2 + i_2
    k_1 = k1 // 2
    i_1 = k1 % 2
    k_2 = k2 // 2
    i_2 = k2 % 2
    # Target index: (2*k_1 + k_2)*4 + (2*i_1 + i_2)
    return (2 * k_1 + k_2) * 4 + (2 * i_1 + i_2)

def _decomp_4x4x4():
    U2, V2, W2 = _strassen_2x2x2()
    U4 = _kron_with_permutation(U2, U2, _perm_u)
    V4 = _kron_with_permutation(V2, V2, _perm_v)
    W4 = _kron_with_permutation(W2, W2, _perm_w)
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS
    # Decompose into 4x4x4 (rank 49), 4x4x4 (rank 49), 4x4x1 (rank 16)
    U4, V4, W4 = _decomp_4x4x4()
    
    rank1 = 49
    rank2 = 49
    rank3 = 16
    total_rank = rank1 + rank2 + rank3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: Columns 0-3 of B
    for i in range(n * m):
        for r in range(rank1):
            U[i][r] = U4[i][r]
    # V for Part 1: Rows 0-3 of B (m=4, p=4)
    # V4 is 4x4 = 16 rows.
    # B is 4x9. Rows 0-3 correspond to B[0:4, 0:4].
    # V row index = j * p + k.
    # For Part 1, k is 0-3.
    for j in range(m):
        for k in range(4):
            for r in range(rank1):
                V[j * p + k][r] = V4[j * 4 + k][r]
    # W for Part 1: Columns 0-3 of C
    for k in range(4):
        for i in range(n):
            for r in range(rank1):
                W[k * n + i][r] = W4[k * 4 + i][r]
                
    # Part 2: Columns 4-7 of B
    offset = rank1
    for i in range(n * m):
        for r in range(rank2):
            U[i][offset + r] = U4[i][r]
    # V for Part 2: Rows 4-7 of B
    for j in range(m):
        for k in range(4):
            for r in range(rank2):
                V[j * p + (4 + k)][offset + r] = V4[j * 4 + k][r]
    # W for Part 2: Columns 4-7 of C
    for k in range(4):
        for i in range(n):
            for r in range(rank2):
                W[k * n + i][offset + r] = W4[k * 4 + i][r]
                
    # Part 3: Column 8 of B (4x4x1)
    offset2 = rank1 + rank2
    # Schoolbook for 4x4x1
    for i in range(n):
        for j in range(m):
            col = offset2 + i * m + j
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            
    return {"U": U, "V": V, "W": W}
