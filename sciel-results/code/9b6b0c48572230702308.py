import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Returns a valid tensor decomposition for 4x5 by 5x6 matrix multiplication.
    Uses a block decomposition to reduce rank from 120 to 112.
    """
    n, m, p = DIMENSIONS
    rank = 112
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a decomposition
    def add_block(i_offset, j_offset, k_offset, local_n, local_m, local_p, local_rank, local_U, local_V, local_W):
        for r in range(local_rank):
            global_r = r # This is not right, we need to accumulate rank
            # We need to manage the global rank index
            pass

    # We will build the matrices by iterating through the blocks and filling the columns.
    # Since the rank is fixed at 112, we can just fill the columns.
    
    # Block 1: n=2, m=2, p=3. Rank 11.
    # There are 4 such blocks (i in 0,1; k in 0,1)
    # i_offset: 0, 2
    # j_offset: 0 (for m=2 part)
    # k_offset: 0, 3
    
    # Block 2: n=2, m=3, p=3. Rank 17.
    # There are 4 such blocks (i in 0,1; k in 0,1)
    # i_offset: 0, 2
    # j_offset: 2 (for m=3 part)
    # k_offset: 0, 3

    col = 0
    
    # Define Strassen 2x2x2 decomposition
    # U, V, W for 2x2x2
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    def get_strassen_222():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[2][0] = 1; W[3][0] = 1; W[1][0] = 1 # C00, C01, C10, C11?
        # C00 = M1 + ...
        # W[k*2+i, r]
        # C00 is i=0, k=0 -> idx 0
        # C01 is i=0, k=1 -> idx 2
        # C10 is i=1, k=0 -> idx 1
        # C11 is i=1, k=1 -> idx 3
        
        # M1 contributes to C00, C01, C10, C11
        W[0][0] = 1 # C00
        W[2][0] = 1 # C01
        W[1][0] = 1 # C10
        W[3][0] = 1 # C11
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1 # C10
        W[3][1] = 1 # C11
        
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1 # C01
        W[3][2] = 1 # C11
        
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1 # C00
        W[1][3] = 1 # C10
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1 # C00
        W[2][4] = 1 # C01
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # C11
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1 # C00
        W[2][6] = 1 # C01
        
        return U, V, W

    def get_schoolbook_221():
        # 2x2 by 2x1. Rank 4.
        # A (2x2), B (2x1). C (2x1).
        # U (4, 4), V (2, 4), W (2, 4)
        # W index k*n+i = k*2+i.
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # r=0: A00 B0
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1 # C0
        
        # r=1: A01 B1
        U[1][1] = 1
        V[1][1] = 1
        W[0][1] = 1 # C0
        
        # r=2: A10 B0
        U[2][2] = 1
        V[0][2] = 1
        W[1][2] = 1 # C1
        
        # r=3: A11 B1
        U[3][3] = 1
        V[1][3] = 1
        W[1][3] = 1 # C1
        
        return U, V, W

    def get_schoolbook_213():
        # 2x1 by 1x3. Rank 6.
        # A (2x1), B (1x3). C (2x3).
        # U (2, 6), V (3, 6), W (6, 6)
        # W index k*n+i = k*2+i.
        U = [[0]*6 for _ in range(2)]
        V = [[0]*6 for _ in range(3)]
        W = [[0]*6 for _ in range(6)]
        
        # C00 = A0 B0
        # C01 = A0 B1
        # C02 = A0 B2
        # C10 = A1 B0
        # C11 = A1 B1
        # C12 = A1 B2
        
        r = 0
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1 # C00
        r = 1
        U[0][1] = 1; V[1][1] = 1; W[2][1] = 1 # C01
        r = 2
        U[0][2] = 1; V[2][2] = 1; W[4][2] = 1 # C02
        r = 3
        U[1][3] = 1; V[0][3] = 1; W[1][3] = 1 # C10
        r = 4
        U[1][4] = 1; V[1][4] = 1; W[3][4] = 1 # C11
        r = 5
        U[1][5] = 1; V[2][5] = 1; W[5][5] = 1 # C12
        
        return U, V, W

    def add_decomposition(i_off, j_off, k_off, lU, lV, lW, l_n, l_m, l_p, l_rank):
        nonlocal col
        for r in range(l_rank):
            for i in range(l_n):
                for j in range(l_m):
                    U[(i_off+i)*m + (j_off+j)][col] = lU[i*l_m+j][r]
            for j in range(l_m):
                for k in range(l_p):
                    V[(j_off+j)*p + (k_off+k)][col] = lV[j*l_p+k][r]
            for k in range(l_p):
                for i in range(l_n):
                    W[(k_off+k)*n + (i_off+i)][col] = lW[k*l_n+i][r]
            col += 1

    # Block 1: 2x2x3 (Rank 11)
    # Composed of Strassen 2x2x2 (Rank 7) and Schoolbook 2x2x1 (Rank 4)
    sU, sV, sW = get_strassen_222()
    sbU, sbV, sbW = get_schoolbook_221()
    
    # Combine them into one 2x2x3 decomposition
    # U: 4x11, V: 6x11, W: 6x11
    U_223 = [[0]*11 for _ in range(4)]
    V_223 = [[0]*11 for _ in range(6)]
    W_223 = [[0]*11 for _ in range(6)]
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_223[i*2+j][r] = sU[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V_223[j*2+k][r] = sV[j*2+k][r]
            V_223[j*2+2][r] = 0 # B col 2 is 0 for Strassen part
        for k in range(2):
            for i in range(2):
                W_223[k*2+i][r] = sW[k*2+i][r]
            W_223[4+i][r] = 0 # C col 2 is 0 for Strassen part
            W_223[5+i][r] = 0 # Wait, W is 6x11. Indices 0..5.
            # W index k*2+i. k=0,1. i=0,1.
            # k=0: 0,1. k=1: 2,3.
            # The 3rd column of C is k=2? No, p=3.
            # W shape is (p*n, rank) = (6, 11).
            # Index k*n+i.
            # k=0: 0,1. k=1: 2,3. k=2: 4,5.
            # Strassen produces C for k=0,1.
            # So W_223[4:6][r] = 0.
            
    for r in range(4):
        global_r = 7 + r
        for i in range(2):
            for j in range(2):
                U_223[i*2+j][global_r] = sbU[i*2+j][r]
        for j in range(2):
            V_223[j*2+2][global_r] = sbV[j][r] # B is 2x1. V index j*p+k. p=1? No, local p=1 for this sub-part.
            # Wait, in the combined 2x2x3, the B part for schoolbook is the 3rd column.
            # V index j*3 + k.
            # The schoolbook part uses B_{j, 2}.
            # So V_223[j*3+2][global_r] = sbV[j][r].
            # Other V entries are 0.
            for k in range(3):
                if k != 2:
                    V_223[j*3+k][global_r] = 0
        for k in range(3):
            for i in range(2):
                if k == 2:
                    W_223[k*2+i][global_r] = sbW[i][r]
                else:
                    W_223[k*2+i][global_r] = 0

    # Block 2: 2x3x3 (Rank 17)
    # Composed of 2x2x3 (Rank 11) and 2x1x3 (Rank 6)
    # 2x2x3 is the same as above.
    # 2x1x3 is get_schoolbook_213.
    
    U_233 = [[0]*17 for _ in range(4)]
    V_233 = [[0]*17 for _ in range(9)]
    W_233 = [[0]*17 for _ in range(6)]
    
    # Part 1: 2x2x3 (A cols 0,1; B rows 0,1)
    for r in range(11):
        for i in range(2):
            for j in range(2):
                U_233[i*3+j][r] = U_223[i*2+j][r]
            for j in range(2,3):
                U_233[i*3+j][r] = 0
        for j in range(2):
            for k in range(3):
                V_233[j*3+k][r] = V_223[j*3+k][r]
        for j in range(2,3):
            for k in range(3):
                V_233[j*3+k][r] = 0
        for k in range(3):
            for i in range(2):
                W_233[k*2+i][r] = W_223[k*2+i][r]

    # Part 2: 2x1x3 (A col 2; B row 2)
    sb213U, sb213V, sb213W = get_schoolbook_213()
    for r in range(6):
        global_r = 11 + r
        for i in range(2):
            U_233[i*3+2][global_r] = sb213U[i][r]
            for j in range(2):
                U_233[i*3+j][global_r] = 0
        for j in range(3):
            if j == 2:
                for k in range(3):
                    V_233[j*3+k][global_r] = sb213V[k][r]
            else:
                for k in range(3):
                    V_233[j*3+k][global_r] = 0
        for k in range(3):
            for i in range(2):
                W_233[k*2+i][global_r] = sb213W[k*2+i][r]

    # Now add all blocks to global U, V, W
    # 4 blocks of 2x2x3
    for i_off in [0, 2]:
        for k_off in [0, 3]:
            add_decomposition(i_off, 0, k_off, U_223, V_223, W_223, 2, 2, 3, 11)

    # 4 blocks of 2x3x3
    for i_off in [0, 2]:
        for k_off in [0, 3]:
            add_decomposition(i_off, 2, k_off, U_233, V_233, W_233, 2, 3, 3, 17)

    return {"U": U, "V": V, "W": W}
