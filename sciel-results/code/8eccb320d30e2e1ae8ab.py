import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n * m, m * p, p * n
    
    # Total rank: 49 (Strassen 4x4x4) + 16 (Schoolbook 4x4x1) = 65
    total_rank = 65
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # --- Part 2: Schoolbook for column 4 (k=4) ---
    # A (4x4) * b (4x1)
    col_offset = 49
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0  # k=4
            W[4 * n + i, col] = 1.0  # k=4, i=row of C
    
    # --- Part 1: Recursive Strassen 4x4x4 ---
    # Level 1: 7 Strassen products, each is a 2x2x2 multiplication
    # Level 2: Each 2x2x2 uses Strassen (7 scalar products)
    
    # Helper functions for index mapping
    def a_idx(bi, u, v):
        """Global index in A for block bi, local (u,v)"""
        i = (bi // 2) * 2 + u
        j = (bi % 2) * 2 + v
        return i * m + j
    
    def b_idx(bi, u, v):
        """Global index in B for block bi, local (u,v) - for 4x4 part (cols 0-3)"""
        j = (bi // 2) * 2 + u  # row of B
        k = (bi % 2) * 2 + v  # col of B (0-3)
        return j * p + k
    
    def c_idx(bi, u, v):
        """Global index in C for block bi, local (u,v)"""
        i = (bi // 2) * 2 + u  # row of C
        k = (bi % 2) * 2 + v  # col of C (0-3)
        return k * n + i
    
    # Strassen 2x2x2 definitions (for the inner level)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each of the 7 outer Strassen products (Mk for k=0..6):
    # Each Mk = A'_k * B'_k where A'_k and B'_k are 2x2 matrices
    # A'_k = sum of coeff * A_block
    # B'_k = sum of coeff * B_block
    
    # Then each 2x2x2 multiplication uses inner Strassen (7 scalar products)
    
    # Outer Strassen definitions
    # alpha[k][b] is coefficient of A_block b in A'_k
    # beta[k][b] is coefficient of B_block b in B'_k
    alpha = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    
    beta = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ]
    
    # Inner Strassen 2x2x2 definitions
    # For a 2x2x2 multiplication: A (2x2) * B (2x2) = C (2x2)
    # Inner Strassen products:
    # N1 = (A00+A11)(B00+B11)
    # N2 = (A10+A11)B00
    # N3 = A00(B01-B11)
    # N4 = A11(B10-B00)
    # N5 = (A00+A01)B11
    # N6 = (A10-A00)(B00+B01)
    # N7 = (A01-A11)(B10+B11)
    
    # Output:
    # C00 = N1 + N4 - N5 + N7
    # C01 = N3 + N5
    # C10 = N2 + N4
    # C11 = N1 - N2 + N3 + N6
    
    # Coefficients for N_l in M_k[u,v]
    # w_k_l_uv is the coefficient of N_l in the (u,v) entry of M_k
    # N_l corresponds to l=0..6
    # u,v in 0..1
    
    # Let's define the linear forms for N_l
    # N0: x00+x11, y00+y11
    # N1: x10+x11, y00
    # N2: x00, y01-y11
    # N3: x11, y10-y00
    # N4: x00+x01, y11
    # N5: x10-x00, y00+y01
    # N6: x01-x11, y10+y11
    
    # We need to map these to the global U and V.
    # For a specific k and l:
    # L_x = sum_{u,v} c_x[u,v] * x[u,v]
    # L_y = sum_{u,v} c_y[u,v] * y[u,v]
    # x[u,v] = sum_b alpha[k][b] * A_b[u,v]
    # y[u,v] = sum_b beta[k][b] * B_b[u,v]
    
    # So U[i*m+j, col] = sum_{u,v,b} c_x[u,v] * alpha[k][b] * delta_{a_idx(b,u,v), i*m+j}
    # V[j*p+k', col] = sum_{u,v,b} c_y[u,v] * beta[k][b] * delta_{b_idx(b,u,v), j*p+k'}
    
    # And W[k''*n+i', col] = sum_{block, u, v} g[block][k] * w_k_l[u,v] * delta_{c_idx(block,u,v), k''*n+i'}
    
    # Let's precompute the linear forms for N_l
    # c_x[l][u,v] and c_y[l][u,v]
    
    c_x = [
        [[1, 0], [0, 1]],  # N0: x00+x11
        [[0, 0], [1, 1]],  # N1: x10+x11
        [[1, 0], [0, 0]],  # N2: x00
        [[0, 0], [0, 1]],  # N3: x11
        [[1, 1], [0, 0]],  # N4: x00+x01
        [[-1, 0], [1, 0]], # N5: x10-x00
        [[0, -1], [0, 1]], # N6: x01-x11 (Wait, x01-x11 is [0,1] - [0,1]? No. x01 is [0,1], x11 is [1,1]. So [0,1]-[1,1] = [-1,0]? No.
                           # x01 is u=0,v=1. x11 is u=1,v=1.
                           # So c_x[6] should be 1 at (0,1) and -1 at (1,1).
    ]
    # Let's re-verify N6: (A01-A11)(B10+B11).
    # x = A01-A11. x01=1, x11=-1.
    # So c_x[6] = [[0, 1], [0, -1]].
    
    c_x = [
        [[1, 0], [0, 1]],  # N0
        [[0, 0], [1, 1]],  # N1
        [[1, 0], [0, 0]],  # N2
        [[0, 0], [0, 1]],  # N3
        [[1, 1], [0, 0]],  # N4
        [[-1, 0], [1, 0]], # N5
        [[0, 1], [0, -1]], # N6
    ]
    
    c_y = [
        [[1, 0], [0, 1]],  # N0: y00+y11
        [[1, 0], [0, 0]],  # N1: y00
        [[0, 1], [0, -1]], # N2: y01-y11
        [[0, -1], [1, 0]], # N3: y10-y00
        [[0, 0], [0, 1]],  # N4: y11
        [[1, 1], [0, 0]],  # N5: y00+y01
        [[0, 1], [1, 0]],  # N6: y10+y11
    ]
    
    # Coefficients of N_l in M_k[u,v]
    # M_k is the result of the 2x2x2 multiplication.
    # C00 = N0 + N3 - N4 + N6
    # C01 = N2 + N4
    # C10 = N1 + N3
    # C11 = N0 - N1 + N2 + N5
    
    w_k_l = [
        [[1, 0, 0, 1, -1, 0, 1],  # u=0,v=0 (C00)
         [0, 0, 1, 0, 1, 0, 0]],  # u=0,v=1 (C01)
        [[0, 1, 0, 1, 0, 0, 0],   # u=1,v=0 (C10)
         [1, -1, 1, 0, 0, 1, 0]]  # u=1,v=1 (C11)
    ]
    # This w_k_l is the same for all k, because the inner Strassen is the same.
    
    # Coefficients of M_k in C_block
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # g[block][k]
    # block 0: C00, block 1: C01, block 2: C10, block 3: C11
    g = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 coeffs for M0..M6
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # Now construct U, V, W for the 49 Strassen columns
    for k in range(7):
        for l in range(7):
            col = k * 7 + l
            
            # Fill U
            for u in range(2):
                for v in range(2):
                    coeff_x = c_x[l][u][v]
                    if coeff_x != 0:
                        for b in range(4):
                            coeff_a = alpha[k][b]
                            if coeff_a != 0:
                                idx = a_idx(b, u, v)
                                U[idx, col] += coeff_x * coeff_a
            
            # Fill V
            for u in range(2):
                for v in range(2):
                    coeff_y = c_y[l][u][v]
                    if coeff_y != 0:
                        for b in range(4):
                            coeff_b = beta[k][b]
                            if coeff_b != 0:
                                idx = b_idx(b, u, v)
                                V[idx, col] += coeff_y * coeff_b
            
            # Fill W
            for block in range(4):
                coeff_g = g[block][k]
                if coeff_g != 0:
                    for u in range(2):
                        for v in range(2):
                            coeff_w = w_k_l[u][v][l]
                            if coeff_w != 0:
                                idx = c_idx(block, u, v)
                                W[idx, col] += coeff_g * coeff_w
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
