"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    # We construct a rank-22 decomposition.
    # Strategy:
    # Split B (3x4) into B1 (3x2) and B2 (3x2).
    # Compute A * B1 (2x3 * 3x2) using rank 11.
    # Compute A * B2 (2x3 * 3x2) using rank 11.
    # Total rank 22.
    
    # Helper to create a rank-11 decomposition for 2x3 * 3x2
    # A is 2x3, B is 3x2.
    # Split A into A_left (2x2) and A_right (2x1).
    # A_left * B is 2x2 * 3x2? No.
    # A = [A1, A2] where A1 is 2x2, A2 is 2x1.
    # B is 3x2.
    # A * B = A1 * B_top + A2 * B_bottom.
    # B_top is 2x2, B_bottom is 1x2.
    # A1 * B_top is 2x2 * 2x2 (Rank 7 Strassen).
    # A2 * B_bottom is 2x1 * 1x2 (Rank 4 Schoolbook).
    
    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2 * 2x2
        # U: 4x7, V: 4x7, W: 4x7
        # Indices:
        # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1 # C00 += M1, C11 += M1
        
        # M2 = (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1 # C10 += M2, C11 -= M2 (handled by sign in W? No, W is just coeff)
        # Wait, Strassen formulas:
        # C00 = M1 - M2 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's map coefficients to W.
        # W[i*2+j][r] is coeff for C[i,j] from product r.
        
        # Reset W
        for i in range(4):
            for j in range(7):
                W[i][j] = 0
                
        # M1
        W[0][0] = 1  # C00
        W[3][0] = 1  # C11
        
        # M2
        W[2][1] = 1  # C10
        W[3][1] = -1 # C11
        
        # M3 = a(f-h)
        U[0][2] = 1
        V[1][2] = 1; V[2][2] = -1
        W[1][2] = 1  # C01
        W[3][2] = 1  # C11
        
        # M4 = d(g-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1  # C00
        W[2][3] = 1  # C10
        
        # M5 = (a+b)h
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1  # C01
        W[0][4] = -1 # C00
        
        # M6 = (c-a)(e+f)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1  # C11
        
        # M7 = (b-d)(g+h)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1  # C00
        
        return U, V, W

    def get_schoolbook_2x1x2():
        # A is 2x1 (indices 0,1), B is 1x2 (indices 0,1)
        # C is 2x2 (indices 0,1,2,3)
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        U = [[0]*4 for _ in range(2)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(4)]
        
        # r=0: A0 * B0 -> C00
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # r=1: A0 * B1 -> C01
        U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
        # r=2: A1 * B0 -> C10
        U[1][2] = 1; V[0][2] = 1; W[2][2] = 1
        # r=3: A1 * B1 -> C11
        U[1][3] = 1; V[1][3] = 1; W[3][3] = 1
        
        return U, V, W

    # Construct full decomposition
    # A indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # B indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3), 8:(2,0), 9:(2,1), 10:(2,2), 11:(2,3)
    # W indices (C): 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3)
    
    # Part 1: A * B1 (cols 0,1 of B)
    # B1 rows:
    # Row 0: B0, B1
    # Row 1: B4, B5
    # Row 2: B8, B9
    
    # A_split:
    # A1 (cols 0,1): A0, A1, A3, A4
    # A2 (col 2): A2, A5
    
    # Sub-problem 1.1: A1 * B1_top (B rows 0,1)
    # A1 is 2x2. B1_top is 2x2.
    # U11, V11, W11 from Strassen.
    # Map A1 indices to global U:
    # A1[0,0]=A0 (0), A1[0,1]=A1 (1), A1[1,0]=A3 (3), A1[1,1]=A4 (4)
    # Map B1_top indices to global V:
    # B1_top[0,0]=B0 (0), B1_top[0,1]=B1 (1), B1_top[1,0]=B4 (4), B1_top[1,1]=B5 (5)
    # Map C11 (2x2) to global W:
    # C11[0,0]=C0 (0), C11[0,1]=C1 (1), C11[1,0]=C4 (4), C11[1,1]=C5 (5)
    
    U11, V11, W11 = get_strassen_2x2x2()
    
    # Sub-problem 1.2: A2 * B1_bottom (B row 2)
    # A2 is 2x1. B1_bottom is 1x2.
    # A2[0]=A2 (2), A2[1]=A5 (5)
    # B1_bottom[0,0]=B8 (8), B1_bottom[0,1]=B9 (9)
    # C12 (2x2) maps to global W:
    # C12[0,0]=C0 (0), C12[0,1]=C1 (1), C12[1,0]=C4 (4), C12[1,1]=C5 (5)
    
    U12, V12, W12 = get_schoolbook_2x1x2()
    
    # Part 2: A * B2 (cols 2,3 of B)
    # B2 rows:
    # Row 0: B2, B3
    # Row 1: B6, B7
    # Row 2: B10, B11
    
    # Sub-problem 2.1: A1 * B2_top
    # A1 same as above.
    # B2_top[0,0]=B2 (2), B2_top[0,1]=B3 (3), B2_top[1,0]=B6 (6), B2_top[1,1]=B7 (7)
    # C21 (2x2) maps to global W:
    # C21[0,0]=C2 (2), C21[0,1]=C3 (3), C21[1,0]=C6 (6), C21[1,1]=C7 (7)
    
    U21, V21, W21 = get_strassen_2x2x2()
    
    # Sub-problem 2.2: A2 * B2_bottom
    # A2 same as above.
    # B2_bottom[0,0]=B10 (10), B2_bottom[0,1]=B11 (11)
    # C22 (2x2) maps to global W:
    # C22[0,0]=C2 (2), C22[0,1]=C3 (3), C22[1,0]=C6 (6), C22[1,1]=C7 (7)
    
    U22, V22, W22 = get_schoolbook_2x1x2()
    
    # Assemble
    rank = 7 + 4 + 7 + 4
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(8)]
    
    # Block 1: Strassen A1*B1_top (Rank 7)
    # Offset 0
    a_map = [0, 1, 3, 4]
    b_map = [0, 1, 4, 5]
    w_map = [0, 1, 4, 5]
    for i in range(4):
        for r in range(7):
            U[a_map[i]][r] = U11[i][r]
            V[b_map[i]][r] = V11[i][r]
            W[w_map[i]][r] = W11[i][r]
            
    # Block 2: Schoolbook A2*B1_bottom (Rank 4)
    # Offset 7
    a_map2 = [2, 5]
    b_map2 = [8, 9]
    w_map2 = [0, 1, 4, 5]
    for i in range(2):
        for r in range(4):
            U[a_map2[i]][7+r] = U12[i][r]
            V[b_map2[i]][7+r] = V12[i][r]
    for i in range(4):
        for r in range(4):
            W[w_map2[i]][7+r] = W12[i][r]
            
    # Block 3: Strassen A1*B2_top (Rank 7)
    # Offset 11
    a_map3 = [0, 1, 3, 4]
    b_map3 = [2, 3, 6, 7]
    w_map3 = [2, 3, 6, 7]
    for i in range(4):
        for r in range(7):
            U[a_map3[i]][11+r] = U21[i][r]
            V[b_map3[i]][11+r] = V21[i][r]
            W[w_map3[i]][11+r] = W21[i][r]
            
    # Block 4: Schoolbook A2*B2_bottom (Rank 4)
    # Offset 18
    a_map4 = [2, 5]
    b_map4 = [10, 11]
    w_map4 = [2, 3, 6, 7]
    for i in range(2):
        for r in range(4):
            U[a_map4[i]][18+r] = U22[i][r]
            V[b_map4[i]][18+r] = V22[i][r]
    for i in range(4):
        for r in range(4):
            W[w_map4[i]][18+r] = W22[i][r]
            
    # Verify
    # n=2, m=3, p=4
    valid = True
    for i in range(2):
        for j in range(3):
            for k in range(4):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*3+j][r] * V[j*4+k][r] * W[k*2+i][r]
                if abs(actual - expected) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to trivial
        rank = 24
        U = [[0] * rank for _ in range(6)]
        V = [[0] * rank for _ in range(12)]
        W = [[0] * rank for _ in range(8)]
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    U[i*3+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
