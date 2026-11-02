import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        # Indices for B entries in block t
        # B is 2x14. Block t is columns 2t, 2t+1.
        # V index = j * p + k
        # j=0 (top row of B), k=2t (e), k=2t+1 (f)
        # j=1 (bottom row of B), k=2t (g), k=2t+1 (h)
        idx_e = 0 * p + 2 * t
        idx_f = 0 * p + 2 * t + 1
        idx_g = 1 * p + 2 * t
        idx_h = 1 * p + 2 * t + 1
        
        # Indices for C entries in block t
        # C is 2x14. Block t is columns 2t, 2t+1.
        # W index = k * n + i
        # i=0 (top row of C), k=2t (C11), k=2t+1 (C12)
        # i=1 (bottom row of C), k=2t (C21), k=2t+1 (C22)
        idx_c11 = 2 * t * n + 0
        idx_c12 = (2 * t + 1) * n + 0
        idx_c21 = 2 * t * n + 1
        idx_c22 = (2 * t + 1) * n + 1
        
        # Strassen products
        # P1: (a+b)(e+h)
        U[0][col] = 1; U[1][col] = 1
        V[idx_e][col] = 1; V[idx_h][col] = 1
        W[idx_c11][col] = 1; W[idx_c22][col] = 1
        col += 1
        
        # P2: (c+d)h
        U[2][col] = 1; U[3][col] = 1
        V[idx_h][col] = 1
        W[idx_c21][col] = 1; W[idx_c22][col] = -1
        col += 1
        
        # P3: a(e-g)
        U[0][col] = 1
        V[idx_e][col] = 1; V[idx_g][col] = -1
        W[idx_c21][col] = 1; W[idx_c22][col] = 1
        col += 1
        
        # P4: d(h-f)
        U[3][col] = 1
        V[idx_h][col] = 1; V[idx_f][col] = -1
        W[idx_c11][col] = 1; W[idx_c22][col] = 1
        col += 1
        
        # P5: (a+b)f
        U[0][col] = 1; U[1][col] = 1
        V[idx_f][col] = 1
        W[idx_c11][col] = -1; W[idx_c12][col] = 1
        col += 1
        
        # P6: (c-a)(e+g)
        U[0][col] = -1; U[2][col] = 1
        V[idx_e][col] = 1; V[idx_g][col] = 1
        W[idx_c12][col] = 1
        col += 1
        
        # P7: (b-d)(h+g)
        U[1][col] = 1; U[3][col] = -1
        V[idx_h][col] = 1; V[idx_g][col] = 1
        W[idx_c11][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
