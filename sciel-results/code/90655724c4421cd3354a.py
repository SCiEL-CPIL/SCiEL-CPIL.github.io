"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=9
    # We use 4 blocks of 2x2 Strassen (rank 7 each) and 1 block of 2x1 Schoolbook (rank 2 each).
    # Total rank = 4 * 7 + 2 = 30.
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val):
        # u_idx: index in U (n*m)
        # v_idx: index in V (m*p)
        # w_idx: index in W (p*n)
        # val: coefficient (1, -1, etc)
        pass # We will just build the matrices directly
    
    col = 0
    
    # Strassen decomposition for 2x2 blocks
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # Indices in U: 0:a00, 1:a01, 2:a10, 3:a11
    # Indices in V for block starting at column offset k_off:
    #   b00: 0*p + k_off
    #   b01: 0*p + k_off + 1
    #   b10: 1*p + k_off
    #   b11: 1*p + k_off + 1
    # Indices in W for block starting at column offset k_off:
    #   C00: k_off*n + 0
    #   C01: (k_off+1)*n + 0
    #   C10: k_off*n + 1
    #   C11: (k_off+1)*n + 1
    
    def add_strassen_block(k_off, start_col):
        nonlocal col
        # M1 = (a00+a01)(b10+b11)
        U[0][col] += 1; U[1][col] += 1
        V[1*p+k_off][col] += 1; V[1*p+k_off+1][col] += 1
        # W: C00 += 1, C11 += 1
        W[k_off*n+0][col] += 1
        W[(k_off+1)*n+1][col] += 1
        col += 1
        
        # M2 = (a10+a11)(b00+b01)
        U[2][col] += 1; U[3][col] += 1
        V[0*p+k_off][col] += 1; V[0*p+k_off+1][col] += 1
        # W: C10 += 1, C11 += -1
        W[k_off*n+1][col] += 1
        W[(k_off+1)*n+1][col] -= 1
        col += 1
        
        # M3 = a00(b00-b10)
        U[0][col] += 1
        V[0*p+k_off][col] += 1; V[1*p+k_off][col] -= 1
        # W: C01 += 1, C11 += 1
        W[(k_off+1)*n+0][col] += 1
        W[(k_off+1)*n+1][col] += 1
        col += 1
        
        # M4 = a11(b10-b11)
        U[3][col] += 1
        V[1*p+k_off][col] += 1; V[1*p+k_off+1][col] -= 1
        # W: C00 += 1, C10 += 1
        W[k_off*n+0][col] += 1
        W[k_off*n+1][col] += 1
        col += 1
        
        # M5 = (a00+a11)b11
        U[0][col] += 1; U[3][col] += 1
        V[1*p+k_off+1][col] += 1
        # W: C00 += -1, C01 += 1
        W[k_off*n+0][col] -= 1
        W[(k_off+1)*n+0][col] += 1
        col += 1
        
        # M6 = (a01-a11)(b10+b11)
        U[1][col] += 1; U[3][col] -= 1
        V[1*p+k_off][col] += 1; V[1*p+k_off+1][col] += 1
        # W: C11 += 1
        W[(k_off+1)*n+1][col] += 1
        col += 1
        
        # M7 = (a10+a00)(b00-b01)
        U[2][col] += 1; U[0][col] += 1
        V[0*p+k_off][col] += 1; V[0*p+k_off+1][col] -= 1
        # W: C00 += 1
        W[k_off*n+0][col] += 1
        col += 1

    # Add 4 Strassen blocks for columns 0-1, 2-3, 4-5, 6-7
    for block in range(4):
        k_off = block * 2
        add_strassen_block(k_off, col)
        
    # Add Schoolbook for column 8
    # C08 = a00*b08 + a01*b18
    # C18 = a10*b08 + a11*b18
    
    # Term 1: a00 * b08
    U[0][col] += 1
    V[0*p+8][col] += 1
    W[8*n+0][col] += 1
    col += 1
    
    # Term 2: a01 * b18
    U[1][col] += 1
    V[1*p+8][col] += 1
    W[8*n+0][col] += 1
    col += 1
    
    # Term 3: a10 * b08
    U[2][col] += 1
    V[0*p+8][col] += 1
    W[8*n+1][col] += 1
    col += 1
    
    # Term 4: a11 * b18
    U[3][col] += 1
    V[1*p+8][col] += 1
    W[8*n+1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
