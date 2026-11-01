import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=4
    # U: (8, rank), V: (16, rank), W: (8, rank)
    
    # Strassen decomposition for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will define the 7 products for a generic block.
    # a,b,c,d are indices into A
    # e,f,g,h are indices into B
    # c11,c12,c21,c22 are indices into C
    
    # Helper to add a block
    def add_strassen_block(U, V, W, a_idx, b_idx, c_idx, col_start):
        # a_idx: [a, b, c, d]
        # b_idx: [e, f, g, h]
        # c_idx: [c11, c12, c21, c22]
        
        a, b, c, d = a_idx
        e, f, g, h = b_idx
        c11, c12, c21, c22 = c_idx
        
        # M1
        col = col_start
        U[a][col] += 1; U[d][col] += 1
        V[e][col] += 1; V[h][col] += 1
        W[c11][col] += 1; W[c22][col] += 1
        # M2
        col = col_start + 1
        U[c][col] += 1; U[d][col] += 1
        V[e][col] += 1
        W[c21][col] += 1; W[c22][col] += -1
        # M3
        col = col_start + 2
        U[a][col] += 1
        V[f][col] += 1; V[h][col] += -1
        W[c12][col] += 1; W[c22][col] += 1
        # M4
        col = col_start + 3
        U[d][col] += 1
        V[g][col] += 1; V[e][col] += -1
        W[c11][col] += 1; W[c21][col] += 1
        # M5
        col = col_start + 4
        U[a][col] += 1; U[b][col] += 1
        V[h][col] += 1
        W[c11][col] += -1; W[c12][col] += 1
        # M6
        col = col_start + 5
        U[c][col] += 1; U[a][col] += -1
        V[e][col] += 1; V[f][col] += 1
        W[c22][col] += 1
        # M7
        col = col_start + 6
        U[b][col] += 1; U[d][col] += -1
        V[g][col] += 1; V[h][col] += 1
        W[c11][col] += 1
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices
    # A: 0..7
    # B: 0..15
    # C: 0..7
    
    # Block 1: A1 B11 -> C1
    # A1: 0,1,4,5
    # B11: 0,1,4,5
    # C1: 0,1,4,5
    add_strassen_block(U, V, W, [0,1,4,5], [0,1,4,5], [0,1,4,5], 0)
    
    # Block 2: A2 B21 -> C1
    # A2: 2,3,6,7
    # B21: 8,9,12,13
    # C1: 0,1,4,5
    add_strassen_block(U, V, W, [2,3,6,7], [8,9,12,13], [0,1,4,5], 7)
    
    # Block 3: A1 B12 -> C2
    # A1: 0,1,4,5
    # B12: 2,3,6,7
    # C2: 2,3,6,7
    add_strassen_block(U, V, W, [0,1,4,5], [2,3,6,7], [2,3,6,7], 14)
    
    # Block 4: A2 B22 -> C2
    # A2: 2,3,6,7
    # B22: 10,11,14,15
    # C2: 2,3,6,7
    add_strassen_block(U, V, W, [2,3,6,7], [10,11,14,15], [2,3,6,7], 21)
    
    return {"U": U, "V": V, "W": W}
