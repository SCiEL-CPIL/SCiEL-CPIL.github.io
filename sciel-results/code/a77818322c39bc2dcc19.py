import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    # Rank 44 decomposition:
    # Split j into j=0 (2x1x8) and j=1,2 (2x2x8)
    # 2x1x8 rank is 16.
    # 2x2x8 is 4 independent 2x2x2 multiplications, each rank 7 (Strassen). Total 28.
    # Total rank = 16 + 28 = 44.
    
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Part 1: 2x1x8 (j=0)
    # U index: i*3 + 0
    # V index: 0*8 + k
    # W index: k*2 + i
    for i in range(n):
        for k in range(p):
            r = i * p + k
            U[i * m + 0][r] = 1.0
            V[0 * p + k][r] = 1.0
            W[k * n + i][r] = 1.0
            
    # Part 2: 2x2x8 (j=1,2)
    # Split k into 4 pairs of 2.
    # For each pair q (0..3), k = 2q + s (s=0,1)
    # Use Strassen for 2x2x2.
    
    # Strassen matrices (local indices i, j_loc in 0..1, s in 0..1)
    # U_loc[i, j_loc], V_loc[j_loc, s], W_loc[s, i]
    
    U_strassen = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[0, 0], [1, 1]], # M6 (c-a) -> row 1 is 1, row 0 is -1? 
                          # M6 = (A10 - A00)(B00 + B01)
                          # U6: A10=1, A00=-1. So [[-1, 0], [1, 0]]
                          # Wait, my previous manual derivation:
                          # M6 = (c-a)(e+f). c=A10, a=A00.
                          # U6[1,0]=1, U6[0,0]=-1.
                          # Let's re-verify U6, V6, W6.
                          # U6 = [[-1, 0], [1, 0]]
                          # V6 = [[1, 1], [0, 0]]
                          # W6 = [[0, 0], [0, 1]]
        [[0, 0], [0, 0]], # Placeholder, will fix
        [[0, 0], [0, 0]]  # Placeholder
    ]
    
    # Correct Strassen definitions:
    # M1: (a+d)(e+h)
    # U1: [[1,0],[0,1]], V1: [[1,0],[0,1]], W1: [[1,0],[0,1]]
    
    # M2: (c+d)e
    # U2: [[0,0],[1,1]], V2: [[1,0],[0,0]], W2: [[0,0],[1,0]]
    
    # M3: a(f-h)
    # U3: [[1,0],[0,0]], V3: [[0,1],[-1,0]], W3: [[0,1],[0,0]]
    
    # M4: d(g-e)
    # U4: [[0,0],[0,1]], V4: [[-1,0],[1,0]], W4: [[1,0],[1,0]]
    
    # M5: (a+b)h
    # U5: [[1,1],[0,0]], V5: [[0,0],[0,1]], W5: [[0,1],[0,0]]
    
    # M6: (c-a)(e+f)
    # U6: [[-1,0],[1,0]], V6: [[1,1],[0,0]], W6: [[0,0],[0,1]]
    
    # M7: (b-d)(g+h)
    # U7: [[0,1],[0,-1]], V7: [[0,0],[1,1]], W7: [[1,0],[0,0]]
    
    U_s = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[0,-1]]
    ]
    V_s = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,1],[-1,0]],
        [[-1,0],[1,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,0],[1,1]]
    ]
    W_s = [
        [[1,0],[0,1]],
        [[0,0],[1,0]],
        [[0,1],[0,0]],
        [[1,0],[1,0]],
        [[0,1],[0,0]],
        [[0,0],[0,1]],
        [[1,0],[0,0]]
    ]
    
    for q in range(4):
        for m_idx in range(7):
            r = 16 + q * 7 + m_idx
            for i in range(2):
                for j_loc in range(2):
                    val_u = U_s[m_idx][i][j_loc]
                    if val_u != 0:
                        U[i * m + (1 + j_loc)][r] += val_u
                        
                for s in range(2):
                    for j_loc in range(2):
                        val_v = V_s[m_idx][j_loc][s]
                        if val_v != 0:
                            V[(1 + j_loc) * p + (2 * q + s)][r] += val_v
                            
                for s in range(2):
                    val_w = W_s[m_idx][s][i]
                    if val_w != 0:
                        W[(2 * q + s) * n + i][r] += val_w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
