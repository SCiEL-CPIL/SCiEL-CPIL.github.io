import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
        n, m, p = DIMENSIONS
        rank = 84
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen coefficients
        # Each entry is a list of (index, coeff) for U, V, W
        # U index: (i_l, j_l)
        # V index: (j_l, k_l)
        # W index: (k_l, i_l)
        
        strassen_terms = [
            # m1
            {
                'U': [(0,0,1), (1,1,1)],
                'V': [(0,0,1), (1,1,1)],
                'W': [(0,0,1), (1,1,1)]
            },
            # m2
            {
                'U': [(1,0,1), (1,1,1)],
                'V': [(0,0,1)],
                'W': [(0,1,1), (1,1,-1)]
            },
            # m3
            {
                'U': [(0,0,1)],
                'V': [(0,1,1), (1,1,-1)],
                'W': [(1,0,1), (1,1,1)]
            },
            # m4
            {
                'U': [(1,1,1)],
                'V': [(1,0,1), (0,0,-1)],
                'W': [(0,0,1), (0,1,1)]
            },
            # m5
            {
                'U': [(0,0,1), (0,1,1)],
                'V': [(1,1,1)],
                'W': [(0,0,-1), (1,0,1)]
            },
            # m6
            {
                'U': [(1,0,1), (0,0,-1)],
                'V': [(0,0,1), (0,1,1)],
                'W': [(1,1,1)]
            },
            # m7
            {
                'U': [(0,1,1), (1,1,-1)],
                'V': [(1,0,1), (1,1,1)],
                'W': [(0,0,1)]
            }
        ]
        
        col = 0
        for g in range(3):
            # Top half of C_g (r=0)
            # A_00 * B_g0
            for s in range(7):
                term = strassen_terms[s]
                r, c, cp = 0, 0, 0
                # U
                for i_l, j_l, val in term['U']:
                    i_g = 2*r + i_l
                    j_g = 2*c + j_l
                    U[i_g * m + j_g][col] = val
                # V
                for j_l, k_l, val in term['V']:
                    j_g = 2*cp + j_l
                    k_g = 2*g + k_l
                    V[j_g * p + k_g][col] = val
                # W
                for k_l, i_l, val in term['W']:
                    k_g = 2*g + k_l
                    i_g = 2*r + i_l
                    W[k_g * n + i_g][col] = val
                col += 1
            
            # A_01 * B_g1
            for s in range(7):
                term = strassen_terms[s]
                r, c, cp = 0, 1, 1
                for i_l, j_l, val in term['U']:
                    i_g = 2*r + i_l
                    j_g = 2*c + j_l
                    U[i_g * m + j_g][col] = val
                for j_l, k_l, val in term['V']:
                    j_g = 2*cp + j_l
                    k_g = 2*g + k_l
                    V[j_g * p + k_g][col] = val
                for k_l, i_l, val in term['W']:
                    k_g = 2*g + k_l
                    i_g = 2*r + i_l
                    W[k_g * n + i_g][col] = val
                col += 1
            
            # Bottom half of C_g (r=1)
            # A_10 * B_g0
            for s in range(7):
                term = strassen_terms[s]
                r, c, cp = 1, 0, 0
                for i_l, j_l, val in term['U']:
                    i_g = 2*r + i_l
                    j_g = 2*c + j_l
                    U[i_g * m + j_g][col] = val
                for j_l, k_l, val in term['V']:
                    j_g = 2*cp + j_l
                    k_g = 2*g + k_l
                    V[j_g * p + k_g][col] = val
                for k_l, i_l, val in term['W']:
                    k_g = 2*g + k_l
                    i_g = 2*r + i_l
                    W[k_g * n + i_g][col] = val
                col += 1
            
            # A_11 * B_g1
            for s in range(7):
                term = strassen_terms[s]
                r, c, cp = 1, 1, 1
                for i_l, j_l, val in term['U']:
                    i_g = 2*r + i_l
                    j_g = 2*c + j_l
                    U[i_g * m + j_g][col] = val
                for j_l, k_l, val in term['V']:
                    j_g = 2*cp + j_l
                    k_g = 2*g + k_l
                    V[j_g * p + k_g][col] = val
                for k_l, i_l, val in term['W']:
                    k_g = 2*g + k_l
                    i_g = 2*r + i_l
                    W[k_g * n + i_g][col] = val
                col += 1

        return {"U": U, "V": V, "W": W}
