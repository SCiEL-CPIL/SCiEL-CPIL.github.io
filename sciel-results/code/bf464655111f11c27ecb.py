"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    
    # 1. Define Strassen factors for 2x2 (Rank 7)
    # U2, V2, W2 are 4x7
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # Row 0 (00)
        [0, 0, 0, 0, 1, 0, 1],  # Row 1 (01)
        [0, 1, 0, 0, 0, 1, 0],  # Row 2 (10)
        [1, 1, 0, 1, 0, 0, -1]  # Row 3 (11)
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0], # Row 0 (00)
        [0, 0, 0, 1, 0, 1, 0], # Row 1 (01)
        [0, 0, 1, -1, 0, 0, 1],# Row 2 (10)
        [1, 0, -1, 0, 1, 0, 1] # Row 3 (11)
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # Row 0 (00) -> C00
        [0, 0, 1, 0, 1, 0, 0],  # Row 1 (01) -> C01
        [0, 1, 0, 1, 0, 0, 0],  # Row 2 (10) -> C10
        [1, -1, 1, 0, 0, 1, 0]  # Row 3 (11) -> C11
    ]
    
    # 2. Define Schoolbook factors for 3x3 (Rank 27)
    # U3, V3, W3 are 9x27
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # U index: i*3 + j
                U3[i*3 + j][col] = 1
                # V index: j*3 + k
                V3[j*3 + k][col] = 1
                # W index: k*3 + i
                W3[k*3 + i][col] = 1
                col += 1
                
    # 3. Construct 6x6 factors (Rank 189)
    n, m, p = DIMENSIONS
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # A: (a, i) x (b, j) -> row 6*(3a+i) + (3b+j)
    # B: (b, j) x (c, k) -> row 6*(3b+j) + (3c+k)
    # C: (c, k) x (a, i) -> row 6*(3c+k) + (3a+i)
    # a,b,c in {0,1}; i,j,k in {0,1,2}
    
    for c2 in range(7):
        for c3 in range(27):
            r = c2 * 27 + c3
            
            # Pre-fetch columns from 2x2 and 3x3 factors
            u2_col = [U2[row][c2] for row in range(4)]
            v2_col = [V2[row][c2] for row in range(4)]
            w2_col = [W2[row][c2] for row in range(4)]
            
            u3_col = [U3[row][c3] for row in range(9)]
            v3_col = [V3[row][c3] for row in range(9)]
            w3_col = [W3[row][c3] for row in range(9)]
            
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        # Indices for 2x2 part
                        u2_idx = 2*a + b
                        v2_idx = 2*b + c
                        w2_idx = 2*c + a
                        
                        u2_val = u2_col[u2_idx]
                        v2_val = v2_col[v2_idx]
                        w2_val = w2_col[w2_idx]
                        
                        # If all are 0, skip inner loop for efficiency (though most are non-zero)
                        if u2_val == 0 or v2_val == 0 or w2_val == 0:
                            continue
                            
                        for i in range(3):
                            for j in range(3):
                                for k in range(3):
                                    # Indices for 3x3 part
                                    u3_idx = 3*i + j
                                    v3_idx = 3*j + k
                                    w3_idx = 3*k + i
                                    
                                    u3_val = u3_col[u3_idx]
                                    v3_val = v3_col[v3_idx]
                                    w3_val = w3_col[w3_idx]
                                    
                                    if u3_val == 0 or v3_val == 0 or w3_val == 0:
                                        continue
                                    
                                    # Calculate linear row indices
                                    row_u = 6 * (3*a + i) + (3*b + j)
                                    row_v = 6 * (3*b + j) + (3*c + k)
                                    row_w = 6 * (3*c + k) + (3*a + i)
                                    
                                    # Calculate values
                                    val_u = u2_val * u3_val
                                    val_v = v2_val * v3_val
                                    val_w = w2_val * w3_val
                                    
                                    U[row_u][r] = val_u
                                    V[row_v][r] = val_v
                                    W[row_w][r] = val_w
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
