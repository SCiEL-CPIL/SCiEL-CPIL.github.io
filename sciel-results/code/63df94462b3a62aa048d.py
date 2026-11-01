"""Matrix multiplication 2,2,6 - rank-21 decomposition via 3 Strassen blocks."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose M_{2,2,6} as sum of 3 Strassen M_{2,2,2} blocks.
    Rank = 3 * 7 = 21.
    
    U shape: (n*m, rank) = (4, 21)
    V shape: (m*p, rank) = (12, 21)  
    W shape: (p*n, rank) = (12, 21)
    
    W index convention: W[k*n + i, r] where k=col of B (0..5), i=row of A (0..1)
    So W row index = k*2 + i, i.e., W[2k+i, r]
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=6
    rank = 21
    U = [[0.0] * rank for _ in range(n * m)]       # 4 x 21
    V = [[0.0] * rank for _ in range(m * p)]       # 12 x 21
    W = [[0.0] * rank for _ in range(p * n)]       # 12 x 21
    
    # Strassen's 7 products for 2x2 x 2x2:
    # Given A (2x2), B (2x2):
    # A = [[a11, a12], [a21, a22]]  -> U indices: a11=0, a12=1, a21=2, a22=3
    # B = [[b11, b12], [b21, b22]]  -> V indices within block
    #
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    #
    # For block t (t=0,1,2), B columns are [2t, 2t+1]
    # V row index for B element at row j, col k: j*p + k = j*6 + k
    
    # Strassen factors for one 2x2x2 block:
    # U factors (for A elements a11,a12,a21,a22 at positions 0,1,2,3):
    strassen_U = [
        [1, 1, 0, 0],   # M1: a11+a22
        [0, 0, 1, 1],   # M2: a21+a22
        [1, 0, 0, 0],   # M3: a11
        [0, 0, 0, 1],   # M4: a22
        [1, 1, 0, 0],   # M5: a11+a12
        [-1, 0, 1, 0],  # M6: a21-a11
        [0, 1, 0, -1],  # M7: a12-a22
    ]
    # V factors (for B elements b11,b12,b21,b22):
    strassen_V = [
        [1, 0, 1, 0],   # M1: b11+b22
        [1, 0, 0, 0],   # M2: b11
        [0, 1, 0, -1],  # M3: b12-b22
        [0, -1, 1, 0],  # M4: b21-b11
        [0, 0, 0, 1],   # M5: b22
        [1, 1, 0, 0],   # M6: b11+b12
        [0, 0, 1, 1],   # M7: b21+b22
    ]
    # W factors (for C elements c11,c12,c21,c22):
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    strassen_W = [
        [1, 0, 0, 0, 1, 0, 1],   # M1..M7 contributions to c11: m1+m4-m5+m7
        [0, 0, 1, 0, 1, 0, 0],   # c12: m3+m5
        [0, 1, 0, 1, 0, 0, 0],   # c21: m2+m4
        [1, -1, 1, 0, 0, 1, 0],  # c22: m1-m2+m3+m6
    ]
    
    for t in range(3):  # 3 blocks
        base = t * 7  # column offset in U, V, W
        for s in range(7):  # 7 Strassen products
            col = base + s
            # U: A elements
            for i in range(4):
                U[i][col] += strassen_U[s][i]
            # V: B elements for block t (columns 2t, 2t+1)
            for j in range(2):  # row of B
                for k in range(2):  # col within block (0 or 1)
                    V[j * p + (2 * t + k)][col] += strassen_V[s][j * 2 + k]
            # W: C elements for block t (columns 2t, 2t+1)
            # W row index = col_of_B * n + row_of_A = (2t+k)*2 + i
            for c_idx in range(4):  # c11=0, c12=1, c21=2, c22=3
                if c_idx == 0:    # c11: i=0, col=2t
                    wrow = (2 * t) * n + 0
                elif c_idx == 1:  # c12: i=0, col=2t+1
                    wrow = (2 * t + 1) * n + 0
                elif c_idx == 2:  # c21: i=1, col=2t
                    wrow = (2 * t) * n + 1
                else:             # c22: i=1, col=2t+1
                    wrow = (2 * t + 1) * n + 1
                W[wrow][col] += strassen_W[c_idx][s]
    
    # Convert to integers where possible (values are integers)
    U = [[int(round(x)) for x in row] for row in U]
    V = [[int(round(x)) for x in row] for row in V]
    W = [[int(round(x)) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
