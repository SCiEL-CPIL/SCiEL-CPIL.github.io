"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    def strassen_4x4_terms():
        """Return the 49 rank-1 terms for 4x4x4 Strassen decomposition.
        Each term is (u, v, w) where u,v,w are length-16 vectors."""
        
        block_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ({(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(1,1): 1}, {(0,1): 1, (1,0): -1}),
            ({(0,0): 1, (1,0): 1}, {(1,1): 1}),
            ({(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ]
        
        sub_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ({(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(1,1): 1}, {(0,1): 1, (1,0): -1}),
            ({(0,0): 1, (1,0): 1}, {(1,1): 1}),
            ({(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ]
        
        c_block_contribs = [
            {((0,0), 1), ((1,1), 1)},
            {((1,0), 1), ((1,1), -1)},
            {((0,1), 1), ((1,1), 1)},
            {((0,0), 1), ((1,0), 1)},
            {((0,0), -1), ((0,1), 1)},
            {((1,1), 1)},
            {((0,0), 1), ((1,1), 1)},
        ]
        
        c_sub_contribs = [
            {((0,0), 1), ((1,1), 1)},
            {((1,0), 1), ((1,1), -1)},
            {((0,1), 1), ((1,1), 1)},
            {((0,0), 1), ((1,0), 1)},
            {((0,0), -1), ((0,1), 1)},
            {((1,1), 1)},
            {((0,0), 1), ((1,1), 1)},
        ]
        
        all_terms = []
        for t_idx, (a_block_coeffs, b_block_coeffs) in enumerate(block_terms):
            for s_idx, (a_sub_coeffs, b_sub_coeffs) in enumerate(sub_terms):
                u = [0] * 16
                for (bi, bj), ca in a_block_coeffs.items():
                    for (si, sj), cs in a_sub_coeffs.items():
                        idx = (2*bi + si) * 4 + (2*bj + sj)
                        u[idx] += ca * cs
                
                v = [0] * 16
                for (bi, bj), cb in b_block_coeffs.items():
                    for (si, sj), cs in b_sub_coeffs.items():
                        idx = (2*bi + si) * 4 + (2*bj + sj)
                        v[idx] += cb * cs
                
                w = [0] * 16
                for (cbi, cbj), cc in c_block_contribs[t_idx]:
                    for (csi, csj), cs in c_sub_contribs[s_idx]:
                        idx = (2*cbi + csi) * 4 + (2*cbj + csj)
                        w[idx] += cc * cs
                
                all_terms.append((u, v, w))
        
        return all_terms
    
    def schoolbook_4x4x1_terms():
        terms = []
        for i in range(4):
            for j in range(4):
                u = [0] * 16
                u[i * 4 + j] = 1
                v = [0] * 4
                v[j] = 1
                w = [0] * 4
                w[i] = 1
                terms.append((u, v, w))
        return terms
    
    strassen_terms = strassen_4x4_terms()
    schoolbook_terms = schoolbook_4x4x1_terms()
    
    total_rank = len(strassen_terms) * 2 + len(schoolbook_terms)
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    col = 0
    for u, v, w in strassen_terms:
        for i in range(16):
            U[i][col] = u[i]
        for j in range(4):
            for k in range(4):
                V[j * 9 + k][col] = v[j * 4 + k]
        for i in range(4):
            for k in range(4):
                W[i * 9 + k][col] = w[i * 4 + k]
        col += 1
    
    for u, v, w in strassen_terms:
        for i in range(16):
            U[i][col] = u[i]
        for j in range(4):
            for k in range(4):
                V[j * 9 + (4 + k)][col] = v[j * 4 + k]
        for i in range(4):
            for k in range(4):
                W[i * 9 + (4 + k)][col] = w[i * 4 + k]
        col += 1
    
    for u, v, w in schoolbook_terms:
        for i in range(16):
            U[i][col] = u[i]
        for j in range(4):
            V[j * 9 + 8][col] = v[j]
        for i in range(4):
            W[i * 9 + 8][col] = w[i]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
