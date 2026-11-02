"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 14 decomposition using two Strassen blocks
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A11-A01)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A (2x2)
    # A00: 0, A01: 1, A10: 2, A11: 3
    
    # Block 1: B cols 0,1 -> C cols 0,1
    # B indices: B00:0, B01:1, B10:4, B11:5
    # C indices: C00:0, C01:2, C10:1, C11:3
    
    # r=0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1 # C00
    W[3][0] = 1 # C11
    
    # r=1: M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1 # C10
    W[3][1] = -1 # C11
    
    # r=2: M3
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1 # C01
    W[3][2] = 1 # C11
    
    # r=3: M4
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1 # C00
    W[1][3] = 1 # C10
    
    # r=4: M5
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1 # C00
    W[2][4] = 1 # C01
    
    # r=5: M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # C11
    
    # r=6: M7
    U[3][6] = 1; U[1][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1 # C00
    
    # Block 2: B cols 2,3 -> C cols 2,3
    # B indices: B00:2, B01:3, B10:6, B11:7
    # C indices: C00:4, C01:6, C10:5, C11:7
    
    # Map B indices: 0->2, 1->3, 4->6, 5->7
    # Map C indices: 0->4, 1->5, 2->6, 3->7
    
    for r in range(7):
        col = r + 7
        # U is same
        for i in range(4):
            U[i][col] = U[i][r]
            
        # V mapping
        v_map = {0: 2, 1: 3, 4: 6, 5: 7}
        for v_idx, val in enumerate(V[r]):
            if val != 0:
                V[v_map[v_idx]][col] = val
                
        # W mapping
        w_map = {0: 4, 1: 5, 2: 6, 3: 7}
        for w_idx, val in enumerate(W[r]):
            if val != 0:
                W[w_map[w_idx]][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
