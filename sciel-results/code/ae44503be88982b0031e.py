import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 8 x 36
    V = [[0] * rank for _ in range(m * p)] # 20 x 36
    W = [[0] * rank for _ in range(p * n)] # 10 x 36

    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        # u_idx: index in U (0..7)
        # v_idx: index in V (0..19)
        # w_idx: index in W (0..9)
        # val: coefficient
        pass

    # We need to fill U, V, W for each of the 36 terms.
    # Terms 0-27: Strassen for 2x4x4
    # Terms 28-35: Schoolbook for 2x4x1

    # Let's define the Strassen terms for a 2x2x2 multiplication
    # Input indices: i in 0..1, j in 0..1, k in 0..1
    # Output indices: k*n + i = 2*k + i
    
    # We have 4 such multiplications:
    # 1. A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # 2. A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2] (added to 1)
    # 3. A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # 4. A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4] (added to 3)

    # Wait, the output of 1 and 2 are added.
    # So we can't just use 4 separate Strassen decompositions and sum the W's?
    # Yes, we can. The W matrix will have entries that are sums of contributions.
    
    # Let's define the Strassen factors for one 2x2x2 multiplication.
    # U is 4x7, V is 4x7, W is 4x7.
    
    def get_strassen_factors():
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # A11 is U[0,0], A22 is U[1,1]
        # B11 is V[0,0], B22 is V[1,1]
        # Term 0:
        U_s[0][0] = 1; U_s[1][0] = 1
        V_s[0][0] = 1; V_s[1][0] = 1
        # W: C11 += M1, C22 += M1
        W_s[0][0] = 1; W_s[3][0] = 1
        
        # M2 = (A21+A22)B11
        # A21 is U[2,0], A22 is U[1,1]
        # B11 is V[0,0]
        # Term 1:
        U_s[2][1] = 1; U_s[1][1] = 1
        V_s[0][1] = 1
        # W: C21 += M2
        W_s[2][1] = 1
        
        # M3 = A11(B12-B22)
        # A11 is U[0,0]
        # B12 is V[0,1], B22 is V[1,1]
        # Term 2:
        U_s[0][2] = 1
        V_s[0][2] = 1; V_s[1][2] = -1
        # W: C12 += M3
        W_s[1][2] = 1
        
        # M4 = A22(B21-B11)
        # A22 is U[1,1]
        # B21 is V[1,0], B11 is V[0,0]
        # Term 3:
        U_s[1][3] = 1
        V_s[1][3] = 1; V_s[0][3] = -1
        # W: C11 += M4, C21 += M4
        W_s[0][3] = 1; W_s[2][3] = 1
        
        # M5 = (A11+A12)B22
        # A11 is U[0,0], A12 is U[0,1]
        # B22 is V[1,1]
        # Term 4:
        U_s[0][4] = 1; U_s[0][4] = 1 # Wait, A11+A12 means U[0,0]+U[0,1]? No.
        # A is 2x2. A11 is row 0 col 0. A12 is row 0 col 1.
        # So U[0][4]=1, U[0][4] is wrong. U[0] is row 0.
        # U[0][4] = 1 (for A11), U[0][4] is the same element? No.
        # U is 4x7. Rows are (0,0), (0,1), (1,0), (1,1).
        # So A11 is U[0], A12 is U[1], A21 is U[2], A22 is U[3].
        # Let's re-index.
        # U[0] -> A11
        # U[1] -> A12
        # U[2] -> A21
        # U[3] -> A22
        
        # Let's restart the Strassen factor definition with correct indices.
        
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # A11=U[0], A22=U[3]. B11=V[0], B22=V[3].
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[3][0] = 1 # C11, C22
        
        # M2 = (A21+A22)B11
        # A21=U[2], A22=U[3]. B11=V[0].
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[2][1] = 1 # C21
        
        # M3 = A11(B12-B22)
        # A11=U[0]. B12=V[1], B22=V[3].
        U_s[0][2] = 1
        V_s[1][2] = 1; V_s[3][2] = -1
        W_s[1][2] = 1 # C12
        
        # M4 = A22(B21-B11)
        # A22=U[3]. B21=V[2], B11=V[0].
        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        W_s[0][3] = 1; W_s[2][3] = 1 # C11, C21
        
        # M5 = (A11+A12)B22
        # A11=U[0], A12=U[1]. B22=V[3].
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[1][4] = 1; W_s[0][4] = -1 # C12, C11
        
        # M6 = (A21-A11)(B11+B12)
        # A21=U[2], A11=U[0]. B11=V[0], B12=V[1].
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[3][5] = 1 # C22
        
        # M7 = (A12-A22)(B21+B22)
        # A12=U[1], A22=U[3]. B21=V[2], B22=V[3].
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[0][6] = 1; W_s[3][6] = -1 # C11, C22

        return U_s, V_s, W_s

    U_s, V_s, W_s = get_strassen_factors()

    # Now map to the full 2x4x4 problem.
    # 4 multiplications:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 3. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 4. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]

    # Indices in U (8x):
    # 0: A[0,0], 1: A[0,1], 2: A[0,2], 3: A[0,3]
    # 4: A[1,0], 5: A[1,1], 6: A[1,2], 7: A[1,3]
    
    # Indices in V (20x):
    # 0: B[0,0], 1: B[0,1], 2: B[0,2], 3: B[0,3], 4: B[0,4]
    # 5: B[1,0], 6: B[1,1], 7: B[1,2], 8: B[1,3], 9: B[1,4]
    # 10: B[2,0], 11: B[2,1], 12: B[2,2], 13: B[2,3], 14: B[2,4]
    # 15: B[3,0], 16: B[3,1], 17: B[3,2], 18: B[3,3], 19: B[3,4]

    # Indices in W (10x):
    # 0: C[0,0], 1: C[0,1], 2: C[0,2], 3: C[0,3], 4: C[0,4]
    # 5: C[1,0], 6: C[1,1], 7: C[1,2], 8: C[1,3], 9: C[1,4]

    # Helper to add a block
    def add_block(u_offsets, v_offsets, w_offsets, start_col):
        for c in range(7):
            for r_u, val_u in enumerate(U_s[r_u][c] for r_u in range(4)):
                if val_u != 0:
                    U[u_offsets[r_u]][start_col + c] += val_u
            for r_v, val_v in enumerate(V_s[r_v][c] for r_v in range(4)):
                if val_v != 0:
                    V[v_offsets[r_v]][start_col + c] += val_v
            for r_w, val_w in enumerate(W_s[r_w][c] for r_w in range(4)):
                if val_w != 0:
                    W[w_offsets[r_w]][start_col + c] += val_w

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # U rows: 0,1,4,5 (A[0,0], A[0,1], A[1,0], A[1,1])
    # V rows: 0,1,5,6 (B[0,0], B[0,1], B[1,0], B[1,1])
    # W rows: 0,1,5,6 (C[0,0], C[0,1], C[1,0], C[1,1])
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 1, 5, 6], 0)

    # Block 2: A[0:2, 2:4] * B[2:4, 0:2]
    # U rows: 2,3,6,7 (A[0,2], A[0,3], A[1,2], A[1,3])
    # V rows: 10,11,15,16 (B[2,0], B[2,1], B[3,0], B[3,1])
    # W rows: 0,1,5,6 (C[0,0], C[0,1], C[1,0], C[1,1])
    add_block([2, 3, 6, 7], [10, 11, 15, 16], [0, 1, 5, 6], 7)

    # Block 3: A[0:2, 0:2] * B[0:2, 2:4]
    # U rows: 0,1,4,5
    # V rows: 2,3,7,8 (B[0,2], B[0,3], B[1,2], B[1,3])
    # W rows: 2,3,7,8 (C[0,2], C[0,3], C[1,2], C[1,3])
    add_block([0, 1, 4, 5], [2, 3, 7, 8], [2, 3, 7, 8], 14)

    # Block 4: A[0:2, 2:4] * B[2:4, 2:4]
    # U rows: 2,3,6,7
    # V rows: 12,13,17,18 (B[2,2], B[2,3], B[3,2], B[3,3])
    # W rows: 2,3,7,8
    add_block([2, 3, 6, 7], [12, 13, 17, 18], [2, 3, 7, 8], 21)

    # Now the 2x4x1 part.
    # C[:, 4] = A * B[:, 4]
    # B[:, 4] is B[0,4], B[1,4], B[2,4], B[3,4] -> V indices 4, 9, 14, 19.
    # C[0,4] is W[4], C[1,4] is W[9].
    # Terms:
    # j=0: A[0,0]*B[0,4] -> U[0], V[4], W[4]
    #       A[1,0]*B[0,4] -> U[4], V[4], W[9]
    # j=1: A[0,1]*B[1,4] -> U[1], V[9], W[4]
    #       A[1,1]*B[1,4] -> U[5], V[9], W[9]
    # j=2: A[0,2]*B[2,4] -> U[2], V[14], W[4]
    #       A[1,2]*B[2,4] -> U[6], V[14], W[9]
    # j=3: A[0,3]*B[3,4] -> U[3], V[19], W[4]
    #       A[1,3]*B[3,4] -> U[7], V[19], W[9]

    start_col = 28
    for j in range(4):
        u0 = j # A[0, j]
        u1 = 4 + j # A[1, j]
        v = 5 * j + 4 # B[j, 4]
        
        # Term 1: A[0,j] * B[j,4] -> C[0,4]
        U[u0][start_col] = 1
        V[v][start_col] = 1
        W[4][start_col] = 1
        start_col += 1
        
        # Term 2: A[1,j] * B[j,4] -> C[1,4]
        U[u1][start_col] = 1
        V[v][start_col] = 1
        W[9][start_col] = 1
        start_col += 1

    return {"U": U, "V": V, "W": W}
