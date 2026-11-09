"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition using Strassen for 2x2x2 blocks and Schoolbook for 2x2x1 blocks
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        for u_idx, val in zip(u_indices, [1]*len(u_indices)):
            pass # Placeholder, we will fill directly

    # Slice 1: A[:, 0:2] * B[0:2, :]
    # A1 indices in U: 0, 1, 2, 3
    # B1 indices in V:
    # Col 0: V[0] (B[0,0]), V[6] (B[1,0])
    # Col 1: V[1] (B[0,1]), V[7] (B[1,1])
    # Col 2: V[2] (B[0,2]), V[8] (B[1,2])
    
    # Strassen for A1 * B1[:, 0:2]
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[7][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[7][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = d(g-e)
    U[3][3] = 1
    V[6][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[7][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[6][6] = 1; V[7][6] = 1
    W[0][6] = 1
    
    # Schoolbook for A1 * B1[:, 2:3]
    # C[0,2] += a*i
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # C[0,2] += b*j
    U[1][8] = 1
    V[8][8] = 1
    W[4][8] = 1
    
    # C[1,2] += c*i
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # C[1,2] += d*j
    U[3][10] = 1
    V[8][10] = 1
    W[5][10] = 1
    
    # Slice 2: A[:, 2:4] * B[2:4, :]
    # A2 indices in U: 4, 5, 6, 7
    # B2 indices in V:
    # Col 0: V[12] (B[2,0]), V[18] (B[3,0])
    # Col 1: V[13] (B[2,1]), V[19] (B[3,1])
    # Col 2: V[14] (B[2,2]), V[20] (B[3,2])
    
    # Strassen for A2 * B2[:, 0:2]
    # M1' = (a'+d')(e'+h')
    U[4][11] = 1; U[7][11] = 1
    V[12][11] = 1; V[19][11] = 1
    W[0][11] = 1; W[3][11] = 1
    
    # M2' = (c'+d')e'
    U[6][12] = 1; U[7][12] = 1
    V[12][12] = 1
    W[1][12] = 1; W[3][12] = -1
    
    # M3' = a'(f'-h')
    U[4][13] = 1
    V[13][13] = 1; V[19][13] = -1
    W[2][13] = 1; W[3][13] = 1
    
    # M4' = d'(g'-e')
    U[7][14] = 1
    V[18][14] = 1; V[12][14] = -1
    W[0][14] = 1; W[1][14] = 1
    
    # M5' = (a'+b')h'
    U[4][15] = 1; U[5][15] = 1
    V[19][15] = 1
    W[0][15] = -1; W[2][15] = 1
    
    # M6' = (c'-a')(e'+f')
    U[6][16] = 1; U[4][16] = -1
    V[12][16] = 1; V[13][16] = 1
    W[3][16] = 1
    
    # M7' = (b'-d')(g'+h')
    U[5][17] = 1; U[7][17] = -1
    V[18][17] = 1; V[19][17] = 1
    W[0][17] = 1
    
    # Schoolbook for A2 * B2[:, 2:3]
    # C[0,2] += a'*i'
    U[4][18] = 1
    V[14][18] = 1
    W[4][18] = 1
    
    # C[0,2] += b'*j'
    U[5][19] = 1
    V[20][19] = 1
    W[4][19] = 1
    
    # C[1,2] += c'*i'
    U[6][20] = 1
    V[14][20] = 1
    W[5][20] = 1
    
    # C[1,2] += d'*j'
    U[7][21] = 1
    V[20][21] = 1
    W[5][21] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
