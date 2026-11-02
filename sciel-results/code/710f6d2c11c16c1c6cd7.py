    # EVOLVE-BLOCK-START
    def construct():
        n, m, p = 2, 4, 6
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen coefficients for 2x2x2
        # Each entry is a list of (i, j, k, coeff) for U, V, W respectively?
        # No, U depends on i,j. V on j,k. W on k,i.
        # We define 7 terms.
        # Term t:
        # U_contribs: list of (i, j, coeff)
        # V_contribs: list of (j, k, coeff)
        # W_contribs: list of (k, i, coeff)
        
        strassen_terms = [
            # M1
            {
                'U': [(0,0,1), (1,1,1)],
                'V': [(0,0,1), (1,1,1)],
                'W': [(0,0,1), (1,1,1)]
            },
            # M2
            {
                'U': [(1,0,1), (1,1,1)],
                'V': [(0,0,1)],
                'W': [(0,1,1)]
            },
            # M3
            {
                'U': [(0,0,1)],
                'V': [(0,1,1), (1,1,-1)],
                'W': [(0,1,1), (1,1,1)]
            },
            # M4
            {
                'U': [(1,1,1)],
                'V': [(1,0,1), (0,0,-1)],
                'W': [(0,0,1), (1,0,1)]
            },
            # M5
            {
                'U': [(0,0,1), (0,1,1)],
                'V': [(1,1,1)],
                'W': [(0,0,-1), (0,1,1)]
            },
            # M6
            {
                'U': [(1,0,1), (0,0,-1)],
                'V': [(0,0,1), (0,1,1)],
                'W': [(1,1,1)]
            },
            # M7
            {
                'U': [(0,1,1), (1,1,-1)],
                'V': [(1,0,1), (1,1,1)],
                'W': [(0,0,1), (1,1,1)]
            }
        ]
        
        # Mapping:
        # j_global = 2 * j_grp + j_local
        # k_global = 2 * k_grp + k_local
        # i_global = i_local (since n=2, no splitting needed for i)
        
        for j_grp in range(2):
            for k_grp in range(3):
                base_col = (j_grp * 3 + k_grp) * 7
                for t in range(7):
                    col = base_col + t
                    term = strassen_terms[t]
                    
                    for (i_l, j_l, c) in term['U']:
                        i_g = i_l
                        j_g = 2 * j_grp + j_l
                        U[i_g * m + j_g][col] += c
                        
                    for (j_l, k_l, c) in term['V']:
                        j_g = 2 * j_grp + j_l
                        k_g = 2 * k_grp + k_l
                        V[j_g * p + k_g][col] += c
                        
                    for (k_l, i_l, c) in term['W']:
                        k_g = 2 * k_grp + k_l
                        i_g = i_l
                        W[k_g * n + i_g][col] += c
                        
        return {'U': U, 'V': V, 'W': W}
    # EVOLVE-BLOCK-END
    