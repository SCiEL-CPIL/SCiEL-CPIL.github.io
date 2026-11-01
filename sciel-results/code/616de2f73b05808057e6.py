"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_strassen_block(b_start, c_start):
        nonlocal col
        # B indices: j*p + k. j in {0,1}, k in {b_start, b_start+1}
        # V0 = 0*p + b_start = b_start
        # V1 = 0*p + b_start+1 = b_start+1
        # V2 = 1*p + b_start = p + b_start
        # V3 = 1*p + b_start+1 = p + b_start+1
        
        # C indices: k*n + i. k in {c_start, c_start+1}, i in {0,1}
        # W0 = c_start*n + 0 = c_start*2
        # W1 = c_start*n + 1 = c_start*2 + 1
        # W2 = (c_start+1)*n + 0 = c_start*2 + 2
        # W3 = (c_start+1)*n + 1 = c_start*2 + 3
        
        v0 = b_start
        v1 = b_start + 1
        v2 = p + b_start
        v3 = p + b_start + 1
        
        w0 = c_start * n
        w1 = c_start * n + 1
        w2 = (c_start + 1) * n
        w3 = (c_start + 1) * n + 1
        
        # M1
        U[0][col] = 1; U[3][col] = 1
        V[v0][col] = 1; V[v3][col] = 1
        W[w0][col] = 1; W[w3][col] = 1
        col += 1
        # M2
        U[2][col] = 1; U[3][col] = 1
        V[v0][col] = 1
        W[w2][col] = 1; W[w3][col] = 1
        col += 1
        # M3
        U[0][col] = 1
        V[v1][col] = 1; V[v3][col] = -1
        W[w1][col] = 1
        col += 1
        # M4
        U[3][col] = 1
        V[v2][col] = 1; V[v0][col] = -1
        W[w0][col] = 1; W[w2][col] = 1
        col += 1
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[v3][col] = 1
        W[w0][col] = -1; W[w1][col] = 1
        col += 1
        # M6
        U[2][col] = 1; U[0][col] = -1
        V[v0][col] = 1; V[v1][col] = 1
        W[w3][col] = 1
        col += 1
        # M7
        U[1][col] = 1; U[3][col] = -1
        V[v2][col] = 1; V[v3][col] = 1
        W[w0][col] = 1; W[w3][col] = 1
        col += 1

    def add_standard_block(b_start, c_start):
        nonlocal col
        # B is 2x1. k=b_start.
        # V0 = 0*p + b_start = b_start
        # V1 = 1*p + b_start = p + b_start
        
        # C is 1x2? No, C is 2x1 block?
        # The block is 2x2x1. Output is 2x1.
        # c_start is the column index in C.
        # W0 = c_start*n + 0 = c_start*2
        # W1 = c_start*n + 1 = c_start*2 + 1
        
        v0 = b_start
        v1 = p + b_start
        
        w0 = c_start * n
        w1 = c_start * n + 1
        
        # C0 = A00 B0 + A01 B1
        U[0][col] = 1
        V[v0][col] = 1
        W[w0][col] = 1
        col += 1
        U[1][col] = 1
        V[v1][col] = 1
        W[w0][col] = 1
        col += 1
        # C1 = A10 B0 + A11 B1
        U[2][col] = 1
        V[v0][col] = 1
        W[w1][col] = 1
        col += 1
        U[3][col] = 1
        V[v1][col] = 1
        W[w1][col] = 1
        col += 1

    add_strassen_block(0, 0)
    add_strassen_block(2, 2)
    add_strassen_block(4, 4)
    add_standard_block(6, 6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
