"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    # We use Strassen's algorithm for each of the 6 blocks of 2x2 in the p dimension.
    # p=12, so 6 blocks.
    # Each block uses rank 7.
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Local indices for A (2x2):
    # a: 0, b: 1, c: 2, d: 3
    # Local indices for B block (2x2):
    # e: 0, f: 1, g: 2, h: 3
    # Local indices for C block (2x2):
    # c00: 0, c01: 1, c10: 2, c11: 3
    
    # Coefficients for U (A), V (B), W (C) for each M term
    # Order: [a, b, c, d] for U
    # Order: [e, f, g, h] for V
    # Order: [c00, c01, c10, c11] for W
    
    terms = [
        # M1
        {
            'U': [1, 0, 0, 1],
            'V': [1, 0, 1, 0],
            'W': [1, 0, 0, 1]
        },
        # M2
        {
            'U': [0, 0, 1, 1],
            'V': [1, 0, 0, 0],
            'W': [0, 0, 1, -1] # C10 += M2, C11 -= M2
        },
        # M3
        {
            'U': [1, 0, 0, 0],
            'V': [0, 1, -1, 0],
            'W': [0, 1, 0, 1] # C01 += M3, C11 += M3
        },
        # M4
        {
            'U': [0, 0, 0, 1],
            'V': [-1, 0, 1, 0], # d(g-e) -> e:-1, g:1
            'W': [1, 0, 1, 1] # C00 += M4, C10 += M4, C11 += M4
        },
        # M5
        {
            'U': [1, 1, 0, 0],
            'V': [0, 0, 0, 1],
            'W': [-1, 1, 0, 0] # C00 -= M5, C01 += M5
        },
        # M6
        {
            'U': [-1, 0, 1, 0], # (c-a)
            'V': [1, 1, 0, 0], # (e+f)
            'W': [0, 0, 0, 1] # C11 += M6
        },
        # M7
        {
            'U': [0, 1, 0, -1], # (b-d)
            'V': [0, 0, 1, 1], # (g+h)
            'W': [1, 0, 0, 0] # C00 += M7
        }
    ]
    
    for j in range(6):
        for s in range(7):
            col = j * 7 + s
            t = terms[s]
            
            # U indices are global 0..3
            for i in range(4):
                U[i][col] = t['U'][i]
                
            # V indices
            # e: B[0, 2j] -> row 2j
            # f: B[0, 2j+1] -> row 2j+1
            # g: B[1, 2j] -> row 12+2j
            # h: B[1, 2j+1] -> row 12+2j+1
            v_rows = [2*j, 2*j+1, 12+2*j, 12+2*j+1]
            for i in range(4):
                V[v_rows[i]][col] = t['V'][i]
                
            # W indices
            # C[0, 2j] -> row 2j*2 + 0 = 4j
            # C[1, 2j] -> row 2j*2 + 1 = 4j+1
            # C[0, 2j+1] -> row (2j+1)*2 + 0 = 4j+2
            # C[1, 2j+1] -> row (2j+1)*2 + 1 = 4j+3
            w_rows = [4*j, 4*j+1, 4*j+2, 4*j+3]
            for i in range(4):
                W[w_rows[i]][col] = t['W'][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
