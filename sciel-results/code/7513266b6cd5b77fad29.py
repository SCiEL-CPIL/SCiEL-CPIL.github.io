"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based block decomposition for 2x2 * 2x7.
    # We split the 2x7 matrix B into two 2x2 blocks B_left (cols 0,1) and B_right (cols 2,3),
    # plus a schoolbook block for the remaining 3 columns (cols 4,5,6).
    #
    # For each 2x2 * 2x2 block, we use Strassen's 7 multiplications.
    # For the 2x2 * 2x3 remainder, we use schoolbook (12 multiplications).
    #
    # Total rank: 7 + 7 + 12 = 26.
    #
    # Strassen formulas for C = A * B where A, B are 2x2:
    #   M1 = A00 * (B01 - B11)
    #   M2 = (A00 + A01) * B11
    #   M3 = (A10 + A11) * B00
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A11) * (B00 + B11)
    #   M6 = (A01 - A11) * (B10 + B11)
    #   M7 = (A00 - A10) * (B00 + B01)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 - M6
    #
    # U rows: A00=0, A01=1, A10=2, A11=3
    # V rows for a 2x2 block: B00, B01, B10, B11
    # W rows for a 2x2 block: C00, C01, C10, C11
    
    n, m, p = DIMENSIONS  # 2, 2, 7
    
    # Build the matrices
    # Rank breakdown:
    # Block 0 (cols 0,1): Strassen, 7 mults
    # Block 1 (cols 2,3): Strassen, 7 mults  
    # Block 2 (cols 4,5,6): Schoolbook, 2*2*3=12 mults
    total_rank = 7 + 7 + 12
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 4 x total_rank
    V = [[0.0] * total_rank for _ in range(m * p)]  # 14 x total_rank
    W = [[0.0] * total_rank for _ in range(p * n)]  # 14 x total_rank
    
    col = 0
    
    # Strassen helper: add 7 Strassen terms for a 2x2 * 2x2 block
    # v_base: starting row index in V for this block's B matrix (rows: B00, B01, B10, B11)
    # w_base: starting row index in W for this block's C matrix (rows: C00, C01, C10, C11)
    def add_strassen(v_base, w_base):
        nonlocal col
        # M1 = A00 * (B01 - B11)
        # U row 0 (A00) gets 1
        U[0][col] = 1
        V[v_base + 1][col] = 1   # B01
        V[v_base + 3][col] = -1  # B11
        W[w_base + 0][col] = 1   # C00
        W[w_base + 3][col] = 1   # C11
        col += 1
        
        # M2 = (A00 + A01) * B11
        U[0][col] = 1
        U[1][col] = 1
        V[v_base + 3][col] = 1   # B11
        W[w_base + 2][col] = 1   # C10
        W[w_base + 3][col] = -1  # C11
        col += 1
        
        # M3 = (A10 + A11) * B00
        U[2][col] = 1
        U[3][col] = 1
        V[v_base + 0][col] = 1   # B00
        W[w_base + 1][col] = 1   # C01
        W[w_base + 3][col] = -1  # C11
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[3][col] = 1
        V[v_base + 2][col] = 1   # B10
        V[v_base + 0][col] = -1  # B00
        W[w_base + 0][col] = 1   # C00
        W[w_base + 2][col] = 1   # C10
        col += 1
        
        # M5 = (A00 + A11) * (B00 + B11)
        U[0][col] = 1
        U[3][col] = 1
        V[v_base + 0][col] = 1   # B00
        V[v_base + 3][col] = 1   # B11
        W[w_base + 0][col] = -1  # C00
        W[w_base + 1][col] = 1   # C01
        col += 1
        
        # M6 = (A01 - A11) * (B10 + B11)
        U[1][col] = 1
        U[3][col] = -1
        V[v_base + 2][col] = 1   # B10
        V[v_base + 3][col] = 1   # B11
        W[w_base + 3][col] = -1  # C11
        col += 1
        
        # M7 = (A00 - A10) * (B00 + B01)
        U[0][col] = 1
        U[2][col] = -1
        V[v_base + 0][col] = 1   # B00
        V[v_base + 1][col] = 1   # B01
        W[w_base + 0][col] = 1   # C00
        col += 1
    
    # Block 0: B columns 0,1 -> v_base=0, w_base=0
    add_strassen(0, 0)
    
    # Block 1: B columns 2,3 -> v_base=4, w_base=4
    add_strassen(4, 4)
    
    # Block 2: B columns 4,5,6 -> schoolbook, 12 mults
    # For each A element (4) and each B column in {4,5,6} (3 cols) and each B row (2):
    # C[i, k] += A[i, j] * B[j, k]
    # V row for B[j, k] = j*p + k
    # W row for C[k, i] = k*n + i (trace convention: W[k*n+i, r])
    for i in range(n):      # row of A, 0 or 1
        for j in range(m):  # col of A / row of B, 0 or 1
            for k in range(4, p):  # col of B, 4,5,6
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
