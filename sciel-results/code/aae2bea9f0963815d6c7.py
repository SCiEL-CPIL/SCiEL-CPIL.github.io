"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # Format: (U_dict, V_dict, W_dict)
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def get_global_A(block, li, lj):
        i = li
        j = lj if block in [0, 2] else 2 + lj
        return i * 4 + j
        
    def get_global_B(block, li, lj):
        j = li if block in [0, 2] else 2 + li
        r = lj if block in [0, 1] else 2 + lj
        return j * 4 + r
        
    def get_global_C(block, li, lj):
        i = li
        r = lj if block in [0, 1] else 2 + lj
        return r * 2 + i

    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            u_c, v_c, w_c = terms[s]
            
            for idx, c in u_c.items():
                li, lj = idx // 2, idx % 2
                U[get_global_A(b, li, lj)][col] += c
                
            for idx, c in v_c.items():
                li, lj = idx // 2, idx % 2
                V[get_global_B(b, li, lj)][col] += c
                
            for idx, c in w_c.items():
                li, lj = idx // 2, idx % 2
                W[get_global_C(b, li, lj)][col] += c

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
