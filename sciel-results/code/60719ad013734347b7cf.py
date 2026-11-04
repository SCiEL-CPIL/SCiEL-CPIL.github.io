"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define groups of size 2 and 1
    # n=3: [0,1], [2]
    # m=5: [0,1], [2,3], [4]
    # p=7: [0,1], [2,3], [4,5], [6]
    
    n_groups = [[0, 1], [2]]
    m_groups = [[0, 1], [2, 3], [4]]
    p_groups = [[0, 1], [2, 3], [4, 5], [6]]
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Strassen 2x2x2 coefficients
    # A indices: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # B indices: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # C indices: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 - b11)
    # M4 = a11(b01 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, -1, 1, 0], # M3
        [0, 1, -1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    strassen_w = [
        [1, 0, 0, 1], # M1 contributes to C00 and C11
        [0, 0, 1, -1], # M2 contributes to C10 and C11
        [0, 1, 0, 1], # M3 contributes to C01 and C11
        [1, 0, 1, 0], # M4 contributes to C00 and C10
        [-1, 1, 0, 0], # M5 contributes to C00 and C01
        [0, 0, 0, 1], # M6 contributes to C11
        [1, 0, 0, 0]  # M7 contributes to C00
    ]
    
    # Wait, let's verify the W coefficients.
    # C00 = M1 + M4 - M5 + M7 -> w00 = [1, 0, 0, 1, -1, 0, 1]
    # C01 = M3 + M5 -> w01 = [0, 0, 1, 0, 1, 0, 0]
    # C10 = M2 + M4 -> w10 = [0, 1, 0, 1, 0, 0, 0]
    # C11 = M1 - M2 + M3 + M6 -> w11 = [1, -1, 1, 0, 0, 1, 0]
    
    # So W_r is a vector of length 4 (for C00, C01, C10, C11).
    # W_1 = [1, 0, 0, 1]
    # W_2 = [0, 0, 1, -1]
    # W_3 = [0, 1, 0, 1]
    # W_4 = [1, 0, 1, 0]
    # W_5 = [-1, 1, 0, 0]
    # W_6 = [0, 0, 0, 1]
    # W_7 = [1, 0, 0, 0]
    
    # This matches my strassen_w list above.
    
    # Iterate over groups
    for gn in n_groups:
        for gm in m_groups:
            for gp in p_groups:
                if len(gn) == 2 and len(gm) == 2 and len(gp) == 2:
                    # Strassen block
                    # gn = [i0, i1], gm = [j0, j1], gp = [k0, k1]
                    i0, i1 = gn
                    j0, j1 = gm
                    k0, k1 = gp
                    
                    # Create local indices mapping
                    # A local indices: 0:(i0,j0), 1:(i0,j1), 2:(i1,j0), 3:(i1,j1)
                    # B local indices: 0:(j0,k0), 1:(j0,k1), 2:(j1,k0), 3:(j1,k1)
                    # C local indices: 0:(i0,k0), 1:(i0,k1), 2:(i1,k0), 3:(i1,k1)
                    
                    for r in range(7):
                        u = [0] * (n * m)
                        v = [0] * (m * p)
                        w = [0] * (p * n)
                        
                        # Fill u
                        u[i0 * m + j0] = strassen_u[r][0]
                        u[i0 * m + j1] = strassen_u[r][1]
                        u[i1 * m + j0] = strassen_u[r][2]
                        u[i1 * m + j1] = strassen_u[r][3]
                        
                        # Fill v
                        v[j0 * p + k0] = strassen_v[r][0]
                        v[j0 * p + k1] = strassen_v[r][1]
                        v[j1 * p + k0] = strassen_v[r][2]
                        v[j1 * p + k1] = strassen_v[r][3]
                        
                        # Fill w
                        # W indices are k*n + i
                        w[k0 * n + i0] = strassen_w[r][0]
                        w[k0 * n + i1] = strassen_w[r][2] # Wait, C10 is index 2
                        w[k1 * n + i0] = strassen_w[r][1] # C01 is index 1
                        w[k1 * n + i1] = strassen_w[r][3] # C11 is index 3
                        
                        # Check W indexing:
                        # C00 (i0,k0) -> w[k0*n + i0]
                        # C01 (i0,k1) -> w[k1*n + i0]
                        # C10 (i1,k0) -> w[k0*n + i1]
                        # C11 (i1,k1) -> w[k1*n + i1]
                        
                        # My strassen_w[r] is [C00, C01, C10, C11]
                        # So:
                        w[k0 * n + i0] = strassen_w[r][0]
                        w[k1 * n + i0] = strassen_w[r][1]
                        w[k0 * n + i1] = strassen_w[r][2]
                        w[k1 * n + i1] = strassen_w[r][3]
                        
                        add_term(u, v, w)
                else:
                    # Schoolbook for each unit in the block
                    for i in gn:
                        for j in gm:
                            for k in gp:
                                u = [0] * (n * m)
                                v = [0] * (m * p)
                                w = [0] * (p * n)
                                
                                u[i * m + j] = 1
                                v[j * p + k] = 1
                                w[k * n + i] = 1
                                
                                add_term(u, v, w)
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
