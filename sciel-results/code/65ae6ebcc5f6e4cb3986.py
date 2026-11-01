"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 5 Strassen blocks (7 terms each) + 1 Schoolbook block (4 terms)
    rank = 5 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen terms for a 2x2 block
    # Indices for U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for V (local j, k): j in 0,1, k in 0,1
    # Indices for W (local k, i): k in 0,1, i in 0,1
    
    def add_strassen_block(k0, k1, start_col):
        # k0, k1 are global column indices for the 2x2 block
        # start_col is the starting rank index
        
        # M1 = (a00 + a11)(b00 + b11)
        U[0][start_col] += 1
        U[3][start_col] += 1
        V[0*p + k0][start_col] += 1
        V[1*p + k1][start_col] += 1
        W[k0*n + 0][start_col] += 1
        W[k1*n + 1][start_col] += 1
        start_col += 1
        
        # M2 = (a10 + a11)b00
        U[2][start_col] += 1
        U[3][start_col] += 1
        V[0*p + k0][start_col] += 1
        W[k0*n + 1][start_col] += 1
        W[k1*n + 1][start_col] -= 1
        start_col += 1
        
        # M3 = a00(b01 - b11)
        U[0][start_col] += 1
        V[0*p + k1][start_col] += 1
        V[1*p + k1][start_col] -= 1
        W[k1*n + 0][start_col] += 1
        W[k1*n + 1][start_col] += 1
        start_col += 1
        
        # M4 = a11(b10 - b00)
        U[3][start_col] += 1
        V[1*p + k0][start_col] += 1
        V[0*p + k0][start_col] -= 1
        W[k0*n + 1][start_col] += 1
        W[k1*n + 1][start_col] += 1
        start_col += 1
        
        # M5 = (a00 + a01)b11
        U[0][start_col] += 1
        U[1][start_col] += 1
        V[1*p + k1][start_col] += 1
        W[k0*n + 0][start_col] -= 1
        W[k1*n + 0][start_col] += 1
        start_col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[2][start_col] += 1
        U[0][start_col] -= 1
        V[0*p + k0][start_col] += 1
        V[0*p + k1][start_col] += 1
        W[k1*n + 1][start_col] += 1
        start_col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[1][start_col] += 1
        U[3][start_col] -= 1
        V[1*p + k0][start_col] += 1
        V[1*p + k1][start_col] += 1
        W[k0*n + 0][start_col] += 1
        start_col += 1
        
        return start_col

    def add_schoolbook_block(k0, start_col):
        # k0 is global column index
        # start_col is the starting rank index
        
        # a00 b00
        U[0][start_col] += 1
        V[0*p + k0][start_col] += 1
        W[k0*n + 0][start_col] += 1
        start_col += 1
        
        # a01 b10
        U[1][start_col] += 1
        V[1*p + k0][start_col] += 1
        W[k0*n + 0][start_col] += 1
        start_col += 1
        
        # a10 b00
        U[2][start_col] += 1
        V[0*p + k0][start_col] += 1
        W[k0*n + 1][start_col] += 1
        start_col += 1
        
        # a11 b10
        U[3][start_col] += 1
        V[1*p + k0][start_col] += 1
        W[k0*n + 1][start_col] += 1
        start_col += 1
        
        return start_col

    # 5 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        col = add_strassen_block(k0, k1, col)
        
    # 1 Schoolbook block for column 10
    col = add_schoolbook_block(10, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
