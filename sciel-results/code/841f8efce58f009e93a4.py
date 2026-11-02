"""Matrix multiplication 2x2 by 2x3 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-11 exact decomposition for multiplying a 2x2 matrix by a 2x3 matrix.
    
    Strategy: Use Strassen's algorithm for the first two columns of the output (rank 7)
    and schoolbook for the third column (rank 4). Total rank = 11.
    
    A is 2x2: A = [[a, b], [c, d]]
    B is 2x3: B = [[e, f, g], [h, i, j]]
    C = A*B is 2x3: C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    
    Indices:
    U row: i*m + j, m=2 -> rows 0,1,2,3 for (i,j) = (0,0),(0,1),(1,0),(1,1)
    V row: j*p + k, p=3 -> rows 0,1,2,3,4,5 for (j,k) = (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    W row: k*n + i, n=2 -> rows 0,1,2,3,4,5 for (k,i) = (0,0),(0,1),(1,0),(1,1),(2,0),(2,1)
    
    Strassen for columns 0,1:
    B_sub = [[e, f], [h, i]]
    M1 = a*(e+i)
    M2 = (c+d)*e
    M3 = d*(h-e)
    M4 = a*(i-f)
    M5 = (a+d)*(e+f)
    M6 = (c-a)*(f+i)
    M7 = (b+c)*(e-h)
    
    C[0,0] = M1 + M3 - M5 + M7
    C[0,1] = M4 + M5
    C[1,0] = M2 + M3
    C[1,1] = M1 - M2 + M4 + M6
    
    Schoolbook for column 2:
    P8 = a*g
    P9 = b*j
    P10 = c*g
    P11 = d*j
    
    C[0,2] = P8 + P9
    C[1,2] = P10 + P11
    
    Now build U, V, W matrices.
    U has 4 rows, V has 6 rows, W has 6 rows.
    Each column r corresponds to one rank-1 term.
    
    For each term, we need to specify which U row, V row, and W row have non-zero entries.
    But since Strassen involves linear combinations, each "multiplication" is actually a product
    of a linear combination of A entries and a linear combination of B entries.
    
    In tensor decomposition terms:
    A linear combination sum_i alpha_i * A[i,j] is represented by U[row_i][r] = alpha_i
    A linear combination sum_j beta_j * B[j,k] is represented by V[row_j][r] = beta_j
    The output C[i',k'] gets contribution gamma * (product) via W[row_k',i'][r] = gamma
    
    Let me define the 11 terms:
    
    Term 0 (M1): A-part: a=1 (row 0 of U), B-part: e=1, i=1 (rows 0,3 of V), W: contributes to C[0,0] and C[1,1]
    C[0,0] gets +1, C[1,1] gets +1
    W[0][0] = 1 (k=0,i=0), W[3][1] = 1 (k=1,i=1)
    U[0][0] = 1
    V[0][0] = 1, V[3][0] = 1
    
    Term 1 (M2): A-part: c=1, d=1 (rows 2,3 of U), B-part: e=1 (row 0 of V), W: contributes to C[1,0]
    C[1,0] gets +1
    W[0][1] = 1 (k=0,i=1)
    U[2][1] = 1, U[3][1] = 1
    V[0][1] = 1
    
    Term 2 (M3): A-part: d=1 (row 3 of U), B-part: h=1, e=-1 (rows 3,0 of V), W: contributes to C[0,0] and C[1,0]
    C[0,0] gets +1, C[1,0] gets +1
    W[0][0] += 1, W[0][1] += 1
    U[3][2] = 1
    V[3][2] = 1, V[0][2] = -1
    
    Term 3 (M4): A-part: a=1 (row 0 of U), B-part: i=1, f=-1 (rows 3,1 of V), W: contributes to C[0,1] and C[1,1]
    C[0,1] gets +1, C[1,1] gets +1
    W[2][0] = 1 (k=1,i=0), W[3][1] += 1
    U[0][3] = 1
    V[3][3] = 1, V[1][3] = -1
    
    Term 4 (M5): A-part: a=1, d=1 (rows 0,3 of U), B-part: e=1, f=1 (rows 0,1 of V), W: contributes to C[0,0] and C[0,1]
    C[0,0] gets -1, C[0,1] gets +1
    W[0][0] += -1, W[2][0] += 1
    U[0][4] = 1, U[3][4] = 1
    V[0][4] = 1, V[1][4] = 1
    
    Term 5 (M6): A-part: c=1, a=-1 (rows 2,0 of U), B-part: f=1, i=1 (rows 1,3 of V), W: contributes to C[1,1]
    C[1,1] gets +1
    W[3][1] += 1
    U[2][5] = 1, U[0][5] = -1
    V[1][5] = 1, V[3][5] = 1
    
    Term 6 (M7): A-part: b=1, c=1 (rows 1,2 of U), B-part: e=1, h=-1 (rows 0,3 of V), W: contributes to C[0,0]
    C[0,0] gets +1
    W[0][0] += 1
    U[1][6] = 1, U[2][6] = 1
    V[0][6] = 1, V[3][6] = -1
    
    Term 7 (P8): A-part: a=1 (row 0 of U), B-part: g=1 (row 2 of V), W: contributes to C[0,2]
    C[0,2] gets +1
    W[4][0] = 1 (k=2,i=0)
    U[0][7] = 1
    V[2][7] = 1
    
    Term 8 (P9): A-part: b=1 (row 1 of U), B-part: j=1 (row 5 of V), W: contributes to C[0,2]
    C[0,2] gets +1
    W[4][0] += 1
    U[1][8] = 1
    V[5][8] = 1
    
    Term 9 (P10): A-part: c=1 (row 2 of U), B-part: g=1 (row 2 of V), W: contributes to C[1,2]
    C[1,2] gets +1
    W[5][1] = 1 (k=2,i=1)
    U[2][9] = 1
    V[2][9] = 1
    
    Term 10 (P11): A-part: d=1 (row 3 of U), B-part: j=1 (row 5 of V), W: contributes to C[1,2]
    C[1,2] gets +1
    W[5][1] += 1
    U[3][10] = 1
    V[5][10] = 1
    
    Let me now construct the matrices.
    """
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Term 0 (M1): a*(e+i)
    U[0][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1   # C[0,0]
    W[3][1] = 1   # C[1,1]
    
    # Term 1 (M2): (c+d)*e
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[0][1] = 1   # C[1,0]
    
    # Term 2 (M3): d*(h-e)
    U[3][2] = 1
    V[3][2] = 1
    V[0][2] = -1
    W[0][0] += 1   # C[0,0]
    W[0][1] += 1   # C[1,0]
    
    # Term 3 (M4): a*(i-f)
    U[0][3] = 1
    V[3][3] = 1
    V[1][3] = -1
    W[2][0] = 1   # C[0,1]
    W[3][1] += 1  # C[1,1]
    
    # Term 4 (M5): (a+d)*(e+f)
    U[0][4] = 1
    U[3][4] = 1
    V[0][4] = 1
    V[1][4] = 1
    W[0][0] += -1  # C[0,0]
    W[2][0] += 1   # C[0,1]
    
    # Term 5 (M6): (c-a)*(f+i)
    U[2][5] = 1
    U[0][5] = -1
    V[1][5] = 1
    V[3][5] = 1
    W[3][1] += 1   # C[1,1]
    
    # Term 6 (M7): (b+c)*(e-h)
    U[1][6] = 1
    U[2][6] = 1
    V[0][6] = 1
    V[3][6] = -1
    W[0][0] += 1   # C[0,0]
    
    # Term 7 (P8): a*g
    U[0][7] = 1
    V[2][7] = 1
    W[4][0] = 1   # C[0,2]
    
    # Term 8 (P9): b*j
    U[1][8] = 1
    V[5][8] = 1
    W[4][0] += 1  # C[0,2]
    
    # Term 9 (P10): c*g
    U[2][9] = 1
    V[2][9] = 1
    W[5][1] = 1   # C[1,2]
    
    # Term 10 (P11): d*j
    U[3][10] = 1
    V[5][10] = 1
    W[5][1] += 1  # C[1,2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
