import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # p=16, so 8 blocks of size 2x2
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (b-c)(e+g)
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Define the 7 terms for a single 2x2x2 block
    # Each term is a dict of (i,j): coeff for U, (j,k): coeff for V, (k,i): coeff for W
    # Note: k here is local 0 or 1.
    
    terms = [
        {
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1} # C00, C11
        },
        {
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(0,1): 1, (1,1): -1} # C10, C11
        },
        {
            'U': {(0,0): 1},
            'V': {(0,1): 1, (1,1): -1},
            'W': {(1,0): 1, (1,1): 1} # C01, C11
        },
        {
            'U': {(1,1): 1},
            'V': {(1,0): 1, (0,0): -1},
            'W': {(0,0): 1, (0,1): 1} # C00, C10
        },
        {
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(0,0): -1, (1,0): 1} # C00, C01
        },
        {
            'U': {(1,0): 1, (1,1): -1},
            'V': {(0,1): 1, (1,1): 1},
            'W': {(1,1): 1} # C11
        },
        {
            'U': {(0,1): 1, (1,0): -1},
            'V': {(0,0): 1, (1,0): 1},
            'W': {(0,0): 1} # C00
        }
    ]
    
    for t in range(num_blocks):
        base_col = t * 7
        for r in range(7):
            col = base_col + r
            term = terms[r]
            
            # Fill U
            for (i, j), val in term['U'].items():
                U[i * m + j][col] = val
                
            # Fill V
            for (j, k_local), val in term['V'].items():
                k_global = t * 2 + k_local
                V[j * p + k_global][col] = val
                
            # Fill W
            for (k_local, i), val in term['W'].items():
                k_global = t * 2 + k_local
                W[k_global * n + i][col] = val
                
    return {"U": U, "V": V, "W": W}
