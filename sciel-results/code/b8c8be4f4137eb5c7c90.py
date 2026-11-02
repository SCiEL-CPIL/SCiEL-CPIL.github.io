"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 5 blocks * 7 Strassen steps = 35
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps definitions
    # U coeffs: [a, b, c, d]
    # V coeffs relative to block [e, f, g, h] -> indices [0, 1, 2, 3] in block
    # W coeffs relative to block [C11, C12, C21, C22] -> indices [0, 1, 2, 3] in block
    
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(g-h)
    # M4: d(e-f)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # W contributions:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    steps = [
        # (u_coeffs, v_block_coeffs, w_block_coeffs)
        # u: a,b,c,d
        # v: e,f,g,h
        # w: C11, C12, C21, C22
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    for t in range(5):
        # Block t corresponds to B columns 2t, 2t+1 and C columns 2t, 2t+1
        # V indices for e, f, g, h
        v_e = 2 * t
        v_f = 2 * t + 1
        v_g = 10 + 2 * t
        v_h = 10 + 2 * t + 1
        v_indices = [v_e, v_f, v_g, v_h]
        
        # W indices for C11, C12, C21, C22
        w_c11 = 2 * t
        w_c12 = 2 * t + 1
        w_c21 = 10 + 2 * t
        w_c22 = 10 + 2 * t + 1
        w_indices = [w_c11, w_c12, w_c21, w_c22]
        
        for s in range(7):
            col = t * 7 + s
            u_coeffs, v_coeffs, w_coeffs = steps[s]
            
            # Fill U
            U[0][col] += u_coeffs[0] # a
            U[1][col] += u_coeffs[1] # b
            U[2][col] += u_coeffs[2] # c
            U[3][col] += u_coeffs[3] # d
            
            # Fill V
            for k in range(4):
                if v_coeffs[k] != 0:
                    V[v_indices[k]][col] += v_coeffs[k]
            
            # Fill W
            for k in range(4):
                if w_coeffs[k] != 0:
                    W[w_indices[k]][col] += w_coeffs[k]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
