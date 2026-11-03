"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U: [a, b, c, d] indices 0, 1, 2, 3
    # V: [e, f, g, h] indices 0, 1, 2, 3
    # W: [C11, C21, C12, C22] indices 0, 1, 2, 3 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    # Term 1: (a+d)(e+h) -> C11 += 1, C22 += 1
    strassen_terms = [
        # U, V, W
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    current_rank = 0
    
    # Group 1: A1 * Btop
    # A1 is A[:, 0:2], Btop is B[0:2, :]
    for q in range(4):
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_terms[s]
            r = current_rank
            for i in range(2):
                for j_loc in range(2):
                    if u_loc[j_loc + i*2] != 0: # u_loc is flat [a,b,c,d] -> i*2+j
                        # Wait, u_loc is [a, b, c, d]. Index i*2+j_loc.
                        # Global U index: i * m + j_global. j_global = j_loc.
                        U[i * m + j_loc][r] = u_loc[i * 2 + j_loc]
                        
            for j_loc in range(2):
                for k_loc in range(2):
                    if v_loc[j_loc * 2 + k_loc] != 0:
                        # Global V index: j_global * p + k_global.
                        # j_global = j_loc. k_global = 2*q + k_loc.
                        V[j_loc * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
                        
            for i in range(2):
                for k_loc in range(2):
                    # W index in local: k_loc * 2 + i.
                    # Global W index: k_global * n + i.
                    # k_global = 2*q + k_loc.
                    W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
            
            current_rank += 1
            
    # Group 2: A2 * Bbottom
    # A2 is A[:, 2:4], Bbottom is B[2:4, :]
    for q in range(4):
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_terms[s]
            r = current_rank
            for i in range(2):
                for j_loc in range(2):
                    if u_loc[i * 2 + j_loc] != 0:
                        # Global U index: i * m + j_global. j_global = j_loc + 2.
                        U[i * m + (j_loc + 2)][r] = u_loc[i * 2 + j_loc]
                        
            for j_loc in range(2):
                for k_loc in range(2):
                    if v_loc[j_loc * 2 + k_loc] != 0:
                        # Global V index: j_global * p + k_global.
                        # j_global = j_loc + 2. k_global = 2*q + k_loc.
                        V[(j_loc + 2) * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
                        
            for i in range(2):
                for k_loc in range(2):
                    # Global W index: k_global * n + i.
                    # k_global = 2*q + k_loc.
                    W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
            
            current_rank += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
