import time

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # Rank = 35
    R = 35
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Strassen coefficients
    # P1 = (A00 + A11)(E + H)
    # P2 = (A10 + A11)E
    # P3 = A00(F - H)
    # P4 = A11(G - E)
    # P5 = (A00 + A01)H
    # P6 = (A10 - A00)(E + F)
    # P7 = (A01 - A11)(G + H)
    
    # A indices: 00, 01, 10, 11
    # B blocks: E, F, G, H
    # E: B[0, 0..4]
    # F: B[0, 5..9]
    # G: B[1, 0..4]
    # H: B[1, 5..9]
    
    # coeff_A[s][i][j]
    # coeff_B[s][block] where block is 0:E, 1:F, 2:G, 3:H
    
    coeff_A = [
        [[0,0],[0,0]], # 0
        [[1,0],[0,1]], # 1: A00 + A11
        [[0,0],[1,1]], # 2: A10 + A11
        [[1,0],[0,0]], # 3: A00
        [[0,0],[0,1]], # 4: A11
        [[1,1],[0,0]], # 5: A00 + A01
        [[1,0],[1,0]], # 6: A10 - A00 -> A00:-1, A10:1
        [[0,1],[0,-1]]  # 7: A01 - A11 -> A01:1, A11:-1
    ]
    # Fix P6: A10 - A00. A00 coeff -1, A10 coeff 1.
    coeff_A[6] = [[-1,0],[1,0]]
    # Fix P7: A01 - A11. A01 coeff 1, A11 coeff -1.
    coeff_A[7] = [[0,1],[0,-1]]
    
    coeff_B = [
        [0,0,0,0], # 0
        [1,0,0,1], # 1: E + H
        [1,0,0,0], # 2: E
        [0,1,0,-1], # 3: F - H
        [0,0,1,-1], # 4: G - E
        [0,0,0,1], # 5: H
        [1,1,0,0], # 6: E + F
        [0,0,1,1]  # 7: G + H
    ]
    
    # Output coefficients coeff_out[i][k][s]
    # i: 0,1
    # k: 0..9
    # s: 1..7
    
    coeff_out = [[[0]*8 for _ in range(p)] for _ in range(n)]
    
    # k in 0..4 (Block 0)
    for k in range(5):
        # C00 = P1 + P4 - P5 + P7
        coeff_out[0][k][1] = 1
        coeff_out[0][k][4] = 1
        coeff_out[0][k][5] = -1
        coeff_out[0][k][7] = 1
        # C10 = P2 + P4
        coeff_out[1][k][2] = 1
        coeff_out[1][k][4] = 1
        
    # k in 5..9 (Block 1)
    for k in range(5, 10):
        # C01 = P3 + P5
        coeff_out[0][k][3] = 1
        coeff_out[0][k][5] = 1
        # C11 = P1 - P2 + P3 + P6
        coeff_out[1][k][1] = 1
        coeff_out[1][k][2] = -1
        coeff_out[1][k][3] = 1
        coeff_out[1][k][6] = 1
        
    # Fill U, V, W
    for s in range(1, 8):
        for l in range(5):
            c = (s - 1) * 5 + l
            
            # U
            for i in range(2):
                for j in range(2):
                    val = coeff_A[s][i][j]
                    if val != 0:
                        U[i * m + j][c] = val
                        
            # V and W
            # Blocks: 0:E, 1:F, 2:G, 3:H
            for b in range(4):
                c_b = coeff_B[s][b]
                if c_b == 0:
                    continue
                
                # Determine k and V index
                if b == 0: # E
                    k = l
                    v_idx = 0 * p + k # 0 * 10 + l
                elif b == 1: # F
                    k = 5 + l
                    v_idx = 0 * p + k
                elif b == 2: # G
                    k = l
                    v_idx = 1 * p + k
                elif b == 3: # H
                    k = 5 + l
                    v_idx = 1 * p + k
                
                V[v_idx][c] = c_b
                
                # W
                for i in range(2):
                    w_val = coeff_out[i][k][s]
                    if w_val != 0:
                        w_idx = k * n + i
                        # Since multiple blocks might map to same k (E and G), 
                        # and w_val is same, we can just assign.
                        # But if we assign, we overwrite. 
                        # Is it possible that w_val is different? No.
                        # Is it possible that we need to sum? 
                        # Term is U * V_b * W. 
                        # Sum over b of U * V_b * W = U * W * Sum(V_b).
                        # So W is just the coefficient.
                        W[w_idx][c] = w_val
                        
    return {"U": U, "V": V, "W": W}
