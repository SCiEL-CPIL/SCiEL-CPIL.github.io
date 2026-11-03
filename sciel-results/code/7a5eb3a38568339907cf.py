<<<<<<< SEARCH
def _strassen_2x4x2():
    """2x4 * 4x2 using two Strassen 2x2x2 blocks (split m=4 into 2+2).
    U: (8, 14), V: (8, 14), W: (4, 14)
    """
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _strassen_2x2x2()
    rank = 14
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(4)]
    
    # First block: rows 0-1 of A (cols 0-1 of B), cols 0-1 of B
    for i in range(4):
        for c in range(7):
            U[i][c] = U1[i][c]
            V[i][c] = V1[i][c]
            W[i][c] = W1[i][c]
    
    # Second block: rows 0-1 of A (cols 2-3 of B), cols 2-3 of B
    # A row index: i*2+j where i in {0,1}, j in {0,1} -> same as before
    # B row index: j*2+k where j in {0,1} for first block (B rows 0-1), j in {2,3} for second
    # Actually B has 4 rows (m=4). Split into B rows 0-1 and B rows 2-3.
    # B index: row*2+col. For first 2x2 block: B rows 0,1 -> indices 0,1,2,3
    # For second 2x2 block: B rows 2,3 -> indices 4,5,6,7
    for i in range(4):
        for c in range(7):
            U[i][7+c] = U2[i][c]
            V[i+4][7+c] = V2[i][c]
            W[i][7+c] = W2[i][c]
    
    return U, V, W
=======
def _strassen_2x4x2():
    """2x4 * 4x2 using two Strassen 2x2x2 blocks (split m=4 into 2+2).
    U: (8, 14), V: (8, 14), W: (4, 14)
    Local U rows: i*4+j. Block 1 uses j=0,1 (rows 0-3). Block 2 uses j=2,3 (rows 4-7).
    Local V rows: j*2+k. Block 1 uses j=0,1 (rows 0-3). Block 2 uses j=2,3 (rows 4-7).
    Local W rows: k*2+i. Both blocks use k=0,1, i=0,1 (rows 0-3).
    """
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _strassen_2x2x2()
    rank = 14
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(4)]
    
    # First block: A cols 0-1, B rows 0-1
    # U rows 0-3, V rows 0-3, W rows 0-3
    for i in range(4):
        for c in range(7):
            U[i][c] = U1[i][c]
            V[i][c] = V1[i][c]
            W[i][c] = W1[i][c]
    
    # Second block: A cols 2-3, B rows 2-3
    # U rows 4-7, V rows 4-7, W rows 0-3
    for i in range(4):
        for c in range(7):
            U[i+4][7+c] = U2[i][c]
            V[i+4][7+c] = V2[i][c]
            W[i][7+c] = W2[i][c]
    
    return U, V, W
>>>>>>> REPLACE

