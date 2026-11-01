"""Matrix multiplication 2x2 by 2x14 - tensor decomposition"""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]
_start = time.time()

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a tensor decomposition for 2x2 by 2x14 matrix multiplication.
    Uses Strassen-style rank-49 construction (7 terms per 2-column block of B).
    """
    n, m, p = DIMENSIONS
    rank = 49  # 7 * (p // 2)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen forms for 2x2 matrix product
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Strassen uses 7 multiplications:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 - M4 + M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M3 + M2 - M7
    
    # U forms (for A components, indexed by i*m+j):
    # 0: a, 1: b, 2: c, 3: d
    U_forms = [
        [1,  0,  0,  1],  # M1: a + d
        [0,  0,  1,  1],  # M2: c + d
        [1,  0,  0,  0],  # M3: a
        [0,  0,  0,  1],  # M4: d
        [1,  1,  0,  0],  # M5: a + b
        [-1, 0,  1,  0],  # M6: -a + c  (c-a)
        [0,  1,  0, -1],  # M7: b - d
    ]
    
    # V forms (for B components, indexed by j*p+k):
    # For block r, columns are 2r, 2r+1
    # B block = [[e, f], [g, h]] at positions (0,2r),(0,2r+1),(1,2r),(1,2r+1)
    # V[0*p+2r] = e, V[0*p+2r+1] = f, V[1*p+2r] = g, V[1*p+2r+1] = h
    
    # W forms (for C components, indexed by k*n+i):
    # C block = [[C00, C01], [C10, C11]] 
    # W[2r*n+0] = C00, W[2r*n+1] = C01, W[(2r+1)*n+0] = C10, W[(2r+1)*n+1] = C11
    
    num_blocks = p // 2  # 7 blocks
    
    for r in range(num_blocks):
        # B indices for this block
        v_e = 0 * p + 2 * r       # B[0, 2r]
        v_f = 0 * p + 2 * r + 1   # B[0, 2r+1]
        v_g = 1 * p + 2 * r       # B[1, 2r]
        v_h = 1 * p + 2 * r + 1   # B[1, 2r+1]
        
        # W indices for this block
        w_c00 = 2 * r * n + 0     # C[2r, 0]
        w_c01 = 2 * r * n + 1     # C[2r, 1]
        w_c10 = (2 * r + 1) * n + 0  # C[2r+1, 0]
        w_c11 = (2 * r + 1) * n + 1  # C[2r+1, 1]
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U (same for all blocks, just A)
            for idx in range(4):
                U[idx][c] = U_forms[s][idx]
            
            # Fill V
            V[v_e][c] = 0
            V[v_f][c] = 0
            V[v_g][c] = 0
            V[v_h][c] = 0
            
            if s == 0:    # M1: (a+d)(e+h)
                V[v_e][c] = 1
                V[v_h][c] = 1
            elif s == 1:  # M2: (c+d)e
                V[v_e][c] = 1
            elif s == 2:  # M3: a(f-h)
                V[v_f][c] = 1
                V[v_h][c] = -1
            elif s == 3:  # M4: d(g-e)
                V[v_e][c] = -1
                V[v_g][c] = 1
            elif s == 4:  # M5: (a+b)h
                V[v_h][c] = 1
            elif s == 5:  # M6: (c-a)(e+f)
                V[v_e][c] = 1
                V[v_f][c] = 1
            elif s == 6:  # M7: (b-d)(g+h)
                V[v_g][c] = 1
                V[v_h][c] = 1
            
            # Fill W
            W[w_c00][c] = 0
            W[w_c01][c] = 0
            W[w_c10][c] = 0
            W[w_c11][c] = 0
            
            if s == 0:    # M1 contributes to C00, C11
                W[w_c00][c] = 1
                W[w_c11][c] = 1
            elif s == 1:  # M2 contributes to C10, C11
                W[w_c10][c] = 1
                W[w_c11][c] = -1
            elif s == 2:  # M3 contributes to C01, C11
                W[w_c01][c] = 1
                W[w_c11][c] = 1
            elif s == 3:  # M4 contributes to C00, C10
                W[w_c00][c] = 1
                W[w_c10][c] = 1
            elif s == 4:  # M5 contributes to C00, C01
                W[w_c00][c] = -1
                W[w_c01][c] = 1
            elif s == 5:  # M6 contributes to C11
                W[w_c11][c] = 1
            elif s == 6:  # M7 contributes to C00
                W[w_c00][c] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
