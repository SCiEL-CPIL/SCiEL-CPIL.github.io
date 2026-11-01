DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 8 for 2x3x2 block
    rank_block = 8
    rank = 3 * rank_block # 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Precompute Strassen 7 coefficients for 2x2x2
    # A = [[a, b], [d, e]], B = [[x, y], [u, v]]
    # P1 = (a+e)(x+u)
    # P2 = (d+e)x
    # P3 = a(u-v)
    # P4 = e(y-x)
    # P5 = (a+b)v
    # P6 = (d-a)(x+y)
    # P7 = (b-e)(u+v)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We need U, V, W for the 7 Strassen terms + 1 Rank1 term.
    # Local indices for U (2x3): 0,1,2,3,4,5
    # 0: a, 1: b, 2: c, 3: d, 4: e, 5: f
    # Local indices for V (3x2): 0,1,2,3,4,5
    # 0: x, 1: y, 2: u, 3: v, 4: s, 5: t
    # Local indices for W (2x2): 0,1,2,3
    # 0: c00, 1: c01, 2: c10, 3: c11
    
    # Term 1: P1
    U[0][0] += 1; U[4][0] += 1
    V[0][0] += 1; V[2][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # Term 2: P2
    U[3][1] += 1; U[4][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] += -1
    
    # Term 3: P3
    U[0][2] += 1
    V[2][2] += 1; V[3][2] += -1
    W[1][2] += 1; W[3][2] += 1
    
    # Term 4: P4
    U[4][3] += 1
    V[1][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[2][3] += 1
    
    # Term 5: P5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1; W[1][4] += 1
    
    # Term 6: P6
    U[3][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # Term 7: P7
    U[1][6] += 1; U[4][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[3][6] += -1
    
    # Term 8: Rank 1 (c*s, c*t, f*s, f*t)
    # u = [0,0,1,0,0,1] (a,b,c,d,e,f) -> c is index 2, f is index 5
    # v = [0,0,0,0,1,1] (x,y,u,v,s,t) -> s is index 4, t is index 5
    # w = [1,1,1,1]
    U[2][7] += 1; U[5][7] += 1
    V[4][7] += 1; V[5][7] += 1
    W[0][7] += 1; W[1][7] += 1; W[2][7] += 1; W[3][7] += 1

    # Now expand to 3 blocks
    # Block 0: V rows 0-5, W rows 0-3
    # Block 1: V rows 6-11, W rows 4-7
    # Block 2: V rows 12-17, W rows 8-11
    
    for block in range(3):
        for local_col in range(8):
            global_col = block * 8 + local_col
            
            # V mapping
            for local_v_row in range(6):
                global_v_row = block * 6 + local_v_row
                V[global_v_row][global_col] = V[local_v_row][local_col]
            
            # W mapping
            for local_w_row in range(4):
                global_w_row = block * 4 + local_w_row
                W[global_w_row][global_col] = W[local_w_row][local_col]
                
            # U is same for all blocks, so we just need to copy the values
            # But U is already filled for cols 0-7.
            # We need to replicate U for cols 8-15 and 16-23.
            if block > 0:
                for u_row in range(6):
                    U[u_row][global_col] = U[u_row][local_col]

    return {"U": U, "V": V, "W": W}
