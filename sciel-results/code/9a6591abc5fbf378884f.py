"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx is list of (row, coeff)
        # v_idx is list of (row, coeff)
        # w_idx is list of (row, coeff)
        pass # We will do it manually for clarity and speed

    col = 0
    
    # --- Block 1: B_1 (cols 0,1) ---
    # V rows: 0, 1, 2, 3
    # W rows: 0, 1, 2, 3
    # U rows: 0, 1, 2, 3
    
    # M1 = (a+d)(e+h)
    U[0][col] += 1; U[3][col] += 1
    V[0][col] += 1; V[3][col] += 1
    W[0][col] += 1; W[3][col] += 1
    col += 1
    
    # M2 = (c+d)e
    U[2][col] += 1; U[3][col] += 1
    V[0][col] += 1
    W[1][col] += 1; W[3][col] += -1
    col += 1
    
    # M3 = a(f-h)
    U[0][col] += 1
    V[1][col] += 1; V[3][col] += -1
    W[2][col] += 1; W[3][col] += 1
    col += 1
    
    # M4 = d(g-e)
    U[3][col] += 1
    V[0][col] += -1; V[2][col] += 1
    W[0][col] += 1; W[1][col] += 1
    col += 1
    
    # M5 = (a+b)h
    U[0][col] += 1; U[1][col] += 1
    V[3][col] += 1
    W[0][col] += -1; W[2][col] += 1
    col += 1
    
    # M6 = (c-a)(e+f)
    U[0][col] += -1; U[2][col] += 1
    V[0][col] += 1; V[1][col] += 1
    W[3][col] += 1
    col += 1
    
    # M7 = (b-d)(g+h)
    U[1][col] += 1; U[3][col] += -1
    V[2][col] += 1; V[3][col] += 1
    W[0][col] += 1
    col += 1
    
    # --- Block 2: B_2 (cols 2,3) ---
    # V rows: 4, 5, 6, 7
    # W rows: 4, 5, 6, 7
    # U rows: 0, 1, 2, 3
    
    # M1' = (a+d)(e'+h')
    U[0][col] += 1; U[3][col] += 1
    V[4][col] += 1; V[7][col] += 1
    W[4][col] += 1; W[7][col] += 1
    col += 1
    
    # M2' = (c+d)e'
    U[2][col] += 1; U[3][col] += 1
    V[4][col] += 1
    W[5][col] += 1; W[7][col] += -1
    col += 1
    
    # M3' = a(f'-h')
    U[0][col] += 1
    V[5][col] += 1; V[7][col] += -1
    W[6][col] += 1; W[7][col] += 1
    col += 1
    
    # M4' = d(g'-e')
    U[3][col] += 1
    V[4][col] += -1; V[6][col] += 1
    W[4][col] += 1; W[5][col] += 1
    col += 1
    
    # M5' = (a+b)h'
    U[0][col] += 1; U[1][col] += 1
    V[7][col] += 1
    W[4][col] += -1; W[6][col] += 1
    col += 1
    
    # M6' = (c-a)(e'+f')
    U[0][col] += -1; U[2][col] += 1
    V[4][col] += 1; V[5][col] += 1
    W[7][col] += 1
    col += 1
    
    # M7' = (b-d)(g'+h')
    U[1][col] += 1; U[3][col] += -1
    V[6][col] += 1; V[7][col] += 1
    W[4][col] += 1
    col += 1
    
    # --- Block 3: B_3 (col 4) ---
    # V rows: 8, 9
    # W rows: 8, 9
    # U rows: 0, 1, 2, 3
    
    # T1 = a y1
    U[0][col] += 1
    V[8][col] += 1
    W[8][col] += 1
    col += 1
    
    # T2 = b y2
    U[1][col] += 1
    V[9][col] += 1
    W[8][col] += 1
    col += 1
    
    # T3 = c y1
    U[2][col] += 1
    V[8][col] += 1
    W[9][col] += 1
    col += 1
    
    # T4 = d y2
    U[3][col] += 1
    V[9][col] += 1
    W[9][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
