"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_indices, v_indices, w_indices, col):
        for u in u_indices:
            U[u][col] = 1
        for v in v_indices:
            V[v][col] = 1
        for w in w_indices:
            W[w][col] = 1
            
    def add_term_signed(u_indices, v_indices, w_indices, col):
        # This is not strictly necessary if we just set 1s and rely on the structure,
        # but Strassen has -1s.
        pass
        
    # Strassen for A1 (2x2) * B11 (2x2) -> C1 (2x2)
    # A1 rows: 0,1,2,3 (i*m+j: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4)
    # B11 rows: 0,1,3,4 (j*p+k: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4)
    # C1 rows: 0,1,2,3 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    # M1: (A00+A11)(B00+B11)
    # U: 0, 4. V: 0, 4. W: 0, 3.
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    # U: 3, 4. V: 0. W: 2, -3.
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A00(B01-B11)
    # U: 0. V: 1, -4. W: 1, 3.
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A11(B10-B00)
    # U: 4. V: 3, -0. W: 0, 2.
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)B11
    # U: 0, 1. V: 4. W: -0, 1.
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # U: 3, -0. V: 0, 1. W: 3.
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # U: 1, -4. V: 3, 4. W: 0.
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # A2 (2x1) * B21 (1x2) -> C1 (2x2)
    # A2 rows: 2, 5 (0*3+2=2, 1*3+2=5)
    # B21 rows: 6, 7 (2*3+0=6, 2*3+1=7)
    # C1 rows: 0, 1, 2, 3
    
    # A02 B20 -> C00 (W0)
    U[2][7] = 1; V[6][7] = 1; W[0][7] = 1
    # A02 B21 -> C01 (W1)
    U[2][8] = 1; V[7][8] = 1; W[1][8] = 1
    # A12 B20 -> C10 (W2)
    U[5][9] = 1; V[6][9] = 1; W[2][9] = 1
    # A12 B21 -> C11 (W3)
    U[5][10] = 1; V[7][10] = 1; W[3][10] = 1
    
    # A1 (2x2) * B12 (2x1) -> C2 (2x1)
    # A1 rows: 0, 1, 3, 4
    # B12 rows: 2, 5 (0*3+2=2, 1*3+2=5)
    # C2 rows: 4, 5 (2*2+0=4, 2*2+1=5)
    
    # A00 B02 -> C20 (W4)
    U[0][11] = 1; V[2][11] = 1; W[4][11] = 1
    # A01 B12 -> C20 (W4)
    U[1][12] = 1; V[5][12] = 1; W[4][12] = 1
    # A10 B02 -> C21 (W5)
    U[3][13] = 1; V[2][13] = 1; W[5][13] = 1
    # A11 B12 -> C21 (W5)
    U[4][14] = 1; V[5][14] = 1; W[5][14] = 1
    
    # A2 (2x1) * B22 (1x1) -> C2 (2x1)
    # A2 rows: 2, 5
    # B22 row: 8 (2*3+2=8)
    # C2 rows: 4, 5
    
    # A02 B22 -> C20 (W4)
    U[2][15] = 1; V[8][15] = 1; W[4][15] = 1
    # A12 B22 -> C21 (W5)
    U[5][16] = 1; V[8][16] = 1; W[5][16] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
