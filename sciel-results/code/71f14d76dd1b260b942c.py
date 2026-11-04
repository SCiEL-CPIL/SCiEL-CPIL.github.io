"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen on 2x2x2 blocks
    # 2 A-blocks (j'=0,1) x 4 B-blocks (k'=0,1,2,3) x 7 Strassen terms
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # c, U_coeffs (i, a), V_coeffs (a, b), W_coeffs (i, b)
    # a in {0,1} corresponds to local j index
    # b in {0,1} corresponds to local k index
    # i in {0,1} corresponds to local i index
    
    strassen_terms = [
        # c=0: M0 = (A00+A11)(B00+B11)
        { 'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (0,1,1), (1,0,1), (1,1,1)] },
        # c=1: M1 = (A10+A11)B00
        { 'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(1,0,1), (1,1,-1)] },
        # c=2: M2 = A00(B01-B11)
        { 'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)] },
        # c=3: M3 = A11(B10-B00)
        { 'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (1,0,1)] },
        # c=4: M4 = (A00+A01)B11
        { 'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (0,1,1)] },
        # c=5: M5 = (A10-A00)(B00+B01)
        { 'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)] },
        # c=6: M6 = (A01-A11)(B10+B11)
        { 'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)] }
    ]
    
    for jp in range(2): # j' block
        for kp in range(4): # k' block
            base_col = (jp * 4 + kp) * 7
            for c in range(7):
                col = base_col + c
                term = strassen_terms[c]
                
                for i, a, val in term['U']:
                    j = jp * 2 + a
                    U[i * m + j][col] = val
                    
                for a, b, val in term['V']:
                    j = jp * 2 + a
                    k = kp * 2 + b
                    V[j * p + k][col] = val
                    
                for i, b, val in term['W']:
                    k = kp * 2 + b
                    W[k * n + i][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
