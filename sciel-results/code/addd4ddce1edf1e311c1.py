import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Winograd 2x2x2 decomposition parameters
    # u_coeffs[r][i][j] = coeff for x_ij in u_r
    # v_coeffs[r][i][j] = coeff for y_ij in v_r
    # w_coeffs[i][j][r] = coeff for M_r in z_ij
    
    u_coeffs = [
        [[1, 0], [0, 1]],  # r=0: x00 + x11
        [[0, 0], [1, 1]],  # r=1: x10 + x11
        [[1, 0], [0, 0]],  # r=2: x00
        [[0, 0], [0, 1]],  # r=3: x11
        [[1, 1], [0, 0]],  # r=4: x00 + x01
        [[-1, 0], [1, 0]], # r=5: x10 - x00
        [[0, 1], [0, -1]]  # r=6: x01 - x11
    ]
    
    v_coeffs = [
        [[1, 1], [0, 0]],  # r=0: y00 + y11 ? No, y00+y11 is not standard. 
                           # Standard: M1 = (x00+x11)(y00+y11)
        # Let's re-verify the standard Winograd 7-term.
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)y00
        # M3 = x00(y10-y11)
        # M4 = x11(y01-y00)
        # M5 = (x00+x01)y11
        # M6 = (x10-x00)(y00+y01)
        # M7 = (x01-x11)(y10+y11)
        
        # My previous v_coeffs:
        # r=0: y00+y11 -> [[1,0],[0,1]]? No, y00 is [0,0], y11 is [1,1].
        # So v0 = [[1,0],[0,1]] is y00+y11. Correct.
        [[1, 0], [0, 1]],  # r=0
        [[1, 0], [0, 0]],  # r=1: y00
        [[0, -1], [1, 0]], # r=2: y10 - y11
        [[-1, 1], [0, 0]], # r=3: y01 - y00
        [[0, 0], [0, 1]],  # r=4: y11
        [[1, 1], [0, 0]],  # r=5: y00 + y01
        [[0, 0], [1, 1]]   # r=6: y10 + y11
    ]
    
    # w_coeffs[i][j][r]
    # z00 = M0 - M2 - M3 + M5
    # z01 = M3 + M4
    # z10 = M1 + M2
    # z11 = M0 - M1 + M4 - M6
    w_coeffs = [
        [
            [1, 0, -1, -1, 0, 1, 0],  # z00
            [0, 0, 0, 1, 1, 0, 0]     # z01
        ],
        [
            [0, 1, 1, 0, 0, 0, 0],    # z10
            [1, -1, 0, 0, 1, 0, -1]   # z11
        ]
    ]
    
    for k in range(4):
        for s in range(2):
            base_col = k * 14 + s * 7
            for r in range(7):
                col = base_col + r
                
                # Fill U
                for i in range(2):
                    for j in range(2):
                        coeff = u_coeffs[r][i][j]
                        if coeff != 0:
                            # A index: row i, col (2*s + j)
                            u_row = i * 4 + (2 * s + j)
                            U[u_row][col] += coeff
                            
                # Fill V
                for i in range(2):
                    for j in range(2):
                        coeff = v_coeffs[r][i][j]
                        if coeff != 0:
                            # B index: row (2*s + i), col (2*k + j)
                            v_row = (2 * s + i) * 8 + (2 * k + j)
                            V[v_row][col] += coeff
                            
                # Fill W
                for i in range(2):
                    for j in range(2):
                        coeff = w_coeffs[i][j][r]
                        if coeff != 0:
                            # C index: row i, col (2*k + j)
                            w_row = (2 * k + j) * 2 + i
                            W[w_row][col] += coeff

    return {"U": U, "V": V, "W": W}
