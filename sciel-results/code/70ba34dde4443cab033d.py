"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    # Block structure:
    # A: 3x3, B: 3x4
    # Row blocks: 2x2 + 1x1
    # Col blocks: 2x2 + 2x2
    #
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (12, rank)
    # W shape: (p*n, rank) = (12, rank)
    #
    # Indexing:
    # U row index: i * m + j
    # V row index: j * p + k
    # W row index: k * n + i
    #
    # Blocks:
    # A11: 2x2 (i=0,1; j=0,1)
    # A12: 2x1 (i=0,1; j=2)
    # A21: 1x2 (i=2; j=0,1)
    # A22: 1x1 (i=2; j=2)
    #
    # B11: 2x2 (j=0,1; k=0,1)
    # B12: 2x2 (j=0,1; k=2,3)
    # B21: 1x2 (j=2; k=0,1)
    # B22: 1x2 (j=2; k=2,3)
    #
    # C blocks:
    # C11: 2x2 (i=0,1; k=0,1) -> rank 7 (Strassen)
    # C12: 2x2 (i=0,1; k=2,3) -> rank 7 (Strassen)
    # C21: 1x2 (i=2; k=0,1) -> rank 4
    # C22: 1x2 (i=2; k=2,3) -> rank 4
    #
    # Total rank: 7 + 7 + 4 + 4 = 22
    
    rank = 22
    U = [[0.0] * rank for _ in range(9)]
    V = [[0.0] * rank for _ in range(12)]
    W = [[0.0] * rank for _ in range(12)]
    
    col = 0
    
    # Strassen helper for 2x2 x 2x2
    def add_strassen(u_rows, v_rows, w_rows):
        """
        u_rows: list of 4 U row indices for A block (row0col0, row0col1, row1col0, row1col1)
        v_rows: list of 4 V row indices for B block (row0col0, row0col1, row1col0, row1col1)
        w_rows: list of 4 W row indices for C block (row0col0, row0col1, row1col0, row1col1)
        Returns: new column index
        """
        nonlocal col
        start = col
        # P1 = (a00+a11)(b00+b11)
        for r in [u_rows[0], u_rows[3]]:
            U[r][col] = 1.0
        for r in [v_rows[0], v_rows[3]]:
            V[r][col] = 1.0
        # C00 += P1
        W[w_rows[0]][col] = 1.0
        # C11 += P1
        W[w_rows[3]][col] = 1.0
        col += 1
        # P2 = (a10+a11)b00
        for r in [u_rows[2], u_rows[3]]:
            U[r][col] = 1.0
        V[v_rows[0]][col] = 1.0
        # C10 += P2
        W[w_rows[2]][col] = 1.0
        # C11 -= P2
        W[w_rows[3]][col] = -1.0
        col += 1
        # P3 = a00(b10-b11)
        U[u_rows[0]][col] = 1.0
        for r in [v_rows[2], v_rows[3]]:
            V[r][col] = 1.0 if r == v_rows[2] else -1.0
        # C01 += P3
        W[w_rows[1]][col] = 1.0
        # C11 += P3
        W[w_rows[3]][col] = 1.0
        col += 1
        # P4 = a11(b01-b00)
        U[u_rows[3]][col] = 1.0
        for r in [v_rows[1], v_rows[0]]:
            V[r][col] = 1.0 if r == v_rows[1] else -1.0
        # C00 += P4
        W[w_rows[0]][col] = 1.0
        # C10 += P4
        W[w_rows[2]][col] = 1.0
        col += 1
        # P5 = (a00+a01)b11
        for r in [u_rows[0], u_rows[1]]:
            U[r][col] = 1.0
        V[v_rows[3]][col] = 1.0
        # C00 -= P5
        W[w_rows[0]][col] = -1.0
        # C01 += P5
        W[w_rows[1]][col] = 1.0
        col += 1
        # P6 = (a10-a00)(b00+b01)
        for r in [u_rows[2], u_rows[0]]:
            U[r][col] = 1.0 if r == u_rows[2] else -1.0
        for r in [v_rows[0], v_rows[1]]:
            V[r][col] = 1.0
        # C11 += P6
        W[w_rows[3]][col] = 1.0
        col += 1
        # P7 = (a01-a11)(b10+b11)
        for r in [u_rows[1], u_rows[3]]:
            U[r][col] = 1.0 if r == u_rows[1] else -1.0
        for r in [v_rows[2], v_rows[3]]:
            V[r][col] = 1.0
        # C00 += P7
        W[w_rows[0]][col] = 1.0
        col += 1
        
        return start
    
    # C11: 2x2 x 2x2 using Strassen
    # A11 rows: i=0,1; j=0,1 -> U indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    # B11 rows: j=0,1; k=0,1 -> V indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # C11 rows: k=0,1; i=0,1 -> W indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    add_strassen([0, 1, 3, 4], [0, 1, 4, 5], [0, 1, 3, 4])
    
    # C12: 2x2 x 2x2 using Strassen
    # A11 rows: same as above
    # B12 rows: j=0,1; k=2,3 -> V indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C12 rows: k=2,3; i=0,1 -> W indices: 2*3+0=6, 2*3+1=7, 3*3+0=9, 3*3+1=10
    add_strassen([0, 1, 3, 4], [2, 3, 6, 7], [6, 7, 9, 10])
    
    # C21: 1x2 x 2x2 -> rank 4
    # A21: i=2; j=0,1 -> U indices: 2*3+0=6, 2*3+1=7
    # B11: j=0,1; k=0,1 -> V indices: 0, 1, 4, 5
    # C21: k=0,1; i=2 -> W indices: 0*3+2=2, 1*3+2=5
    # Terms:
    # a20*b10 -> C20
    # a20*b11 -> C21
    # a21*b10 -> C20
    # a21*b11 -> C21
    U[6][col] = 1.0
    V[0][col] = 1.0
    W[2][col] = 1.0
    col += 1
    U[6][col] = 1.0
    V[1][col] = 1.0
    W[5][col] = 1.0
    col += 1
    U[7][col] = 1.0
    V[0][col] = 1.0
    W[2][col] = 1.0
    col += 1
    U[7][col] = 1.0
    V[1][col] = 1.0
    W[5][col] = 1.0
    col += 1
    
    # C22: 1x2 x 2x2 -> rank 4
    # A21: i=2; j=0,1 -> U indices: 6, 7
    # B12: j=0,1; k=2,3 -> V indices: 2, 3, 6, 7
    # C22: k=2,3; i=2 -> W indices: 2*3+2=8, 3*3+2=11
    U[6][col] = 1.0
    V[2][col] = 1.0
    W[8][col] = 1.0
    col += 1
    U[6][col] = 1.0
    V[3][col] = 1.0
    W[11][col] = 1.0
    col += 1
    U[7][col] = 1.0
    V[2][col] = 1.0
    W[8][col] = 1.0
    col += 1
    U[7][col] = 1.0
    V[3][col] = 1.0
    W[11][col] = 1.0
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
