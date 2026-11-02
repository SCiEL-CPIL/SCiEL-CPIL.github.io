import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen terms
    # u: size 4 (A00, A01, A10, A11)
    # v: size 2x2 (rows 0,1 of B, cols 0,1 of block)
    # w: size 2x2 (rows 0,1 of C, cols 0,1 of block)
    
    terms = []
    
    # M1: (A00 + A11) * (B00 + B11)
    u1 = [1, 0, 0, 1]
    v1 = [[1, 0], [0, 1]]
    w1 = [[1, 0], [0, 1]]
    terms.append((u1, v1, w1))
    
    # M2: (A10 + A11) * B00
    u2 = [0, 0, 1, 1]
    v2 = [[1, 1], [0, 0]]
    w2 = [[0, 0], [1, -1]]
    terms.append((u2, v2, w2))
    
    # M3: A00 * (B01 - B11)
    u3 = [1, 0, 0, 0]
    v3 = [[0, 1], [-1, 0]] # B01 is row 0 col 1. B11 is row 1 col 1.
    # v3[0,1] = 1, v3[1,1] = -1.
    w3 = [[0, 1], [0, 1]] # C01 += M3, C11 += M3
    terms.append((u3, v3, w3))
    
    # M4: A11 * (B10 - B00)
    u4 = [0, 0, 0, 1]
    v4 = [[-1, 0], [1, 0]] # B10 is row 1 col 0. B00 is row 0 col 0.
    w4 = [[1, 0], [1, 0]] # C00 += M4, C10 += M4
    terms.append((u4, v4, w4))
    
    # M5: (A00 + A01) * B11
    u5 = [1, 1, 0, 0]
    v5 = [[0, 0], [0, 1]] # B11 is row 1 col 1.
    w5 = [[-1, 0], [0, 1]] # C00 -= M5, C01 += M5
    terms.append((u5, v5, w5))
    
    # M6: (A10 - A00) * (B00 + B01)
    u6 = [-1, 0, 1, 0]
    v6 = [[1, 1], [0, 0]]
    w6 = [[0, 0], [0, 1]] # C11 += M6
    terms.append((u6, v6, w6))
    
    # M7: (A01 - A11) * (B10 + B11)
    u7 = [0, 1, 0, -1]
    v7 = [[0, 0], [1, 1]]
    w7 = [[1, 0], [0, 0]] # C00 += M7
    terms.append((u7, v7, w7))
    
    col = 0
    for b in range(num_blocks):
        # Global column offset in B and C
        k_start = 2 * b
        
        for u, v, w in terms:
            # Fill U
            for i in range(4):
                U[i, col] = u[i]
                
            # Fill V
            # v[j, l] contributes to V[j * p + (k_start + l), col]
            for j in range(2):
                for l in range(2):
                    v_idx = j * p + (k_start + l)
                    V[v_idx, col] = v[j][l]
                    
            # Fill W
            # w[i, l] contributes to W[(k_start + l) * n + i, col]
            for i in range(2):
                for l in range(2):
                    w_idx = (k_start + l) * n + i
                    W[w_idx, col] = w[i][l]
                    
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
