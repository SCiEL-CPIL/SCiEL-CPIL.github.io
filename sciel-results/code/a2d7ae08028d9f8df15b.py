"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 110 decomposition using 1+4 split with Strassen for 4x4
    total_rank = 110
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A11 * B11 (1x1)
    # A[0,0] * B[0,0] -> C[0,0]
    U[0][col] = 1
    V[0][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. A12 * B21 (1x4 * 4x1)
    # Sum_{j=1..4} A[0,j] * B[j,0] -> C[0,0]
    for j in range(1, 5):
        U[0][col] = 1
        V[j * 5][col] = 1
        W[0][col] = 1
        col += 1
        
    # 3. A11 * B12 (1x1 * 1x4)
    # A[0,0] * B[0,k] -> C[0,k]
    for k in range(1, 5):
        U[0][col] = 1
        V[k][col] = 1
        W[k * 5][col] = 1
        col += 1
        
    # 4. A12 * B22 (1x4 * 4x4)
    # Sum_{j=1..4} A[0,j] * B[j,k] -> C[0,k]
    for j in range(1, 5):
        for k in range(1, 5):
            U[0][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 5][col] = 1
            col += 1
            
    # 5. A21 * B11 (4x1 * 1x1)
    # A[i,0] * B[0,0] -> C[i,0]
    for i in range(1, 5):
        U[i * 5][col] = 1
        V[0][col] = 1
        W[i][col] = 1
        col += 1
        
    # 6. A22 * B21 (4x4 * 4x1)
    # Sum_{j=1..4} A[i,j] * B[j,0] -> C[i,0]
    for i in range(1, 5):
        for j in range(1, 5):
            U[i * 5 + j][col] = 1
            V[j * 5][col] = 1
            W[i][col] = 1
            col += 1
            
    # 7. A21 * B12 (4x1 * 1x4)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(1, 5):
        for k in range(1, 5):
            U[i * 5][col] = 1
            V[k][col] = 1
            W[k * 5 + i][col] = 1
            col += 1
            
    # 8. A22 * B22 (4x4 * 4x4) using Strassen
    # We need to generate the 49 terms for the 4x4 block at indices 1..4
    # and add them to U, V, W at the correct global indices.
    
    def get_strassen_2x2():
        """Returns U, V, W for 2x2x2 multiplication with rank 7."""
        # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        # U: 1 at 0, 3. V: 1 at 0, 3. W: 1 at 0 (C00), 1 at 3 (C11)
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[3][0] = 1
        
        # M2 = (A10+A11)B00
        # U: 1 at 2, 3. V: 1 at 0. W: 1 at 2 (C10), 1 at 3 (C11) with -1
        U2[2][1] = 1; U2[3][1] = 1
        V2[0][1] = 1
        W2[2][1] = 1; W2[3][1] = -1
        
        # M3 = A00(B01-B11)
        # U: 1 at 0. V: 1 at 1, -1 at 3. W: 1 at 1 (C01), 1 at 3 (C11)
        U2[0][2] = 1
        V2[1][2] = 1; V2[3][2] = -1
        W2[1][2] = 1; W2[3][2] = 1
        
        # M4 = A11(B10-B00)
        # U: 1 at 3. V: 1 at 2, -1 at 0. W: 1 at 0 (C00), 1 at 2 (C10)
        U2[3][3] = 1
        V2[2][3] = 1; V2[0][3] = -1
        W2[0][3] = 1; W2[2][3] = 1
        
        # M5 = (A00+A01)B11
        # U: 1 at 0, 1. V: 1 at 3. W: -1 at 0 (C00), 1 at 1 (C01)
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[0][4] = -1; W2[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        # U: 1 at 2, -1 at 0. V: 1 at 0, 1. W: 1 at 3 (C11)
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[1][5] = 1
        W2[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        # U: 1 at 1, -1 at 3. V: 1 at 2, 3. W: 1 at 0 (C00)
        U2[1][6] = 1; U2[3][6] = -1
        V2[2][6] = 1; V2[3][6] = 1
        W2[0][6] = 1
        
        return U2, V2, W2

    U2, V2, W2 = get_strassen_2x2()
    
    # Mapping for 4x4 blocks
    # Block (0,0): rows 0,1 cols 0,1 -> global 0,1,5,6
    # Block (0,1): rows 0,1 cols 2,3 -> global 2,3,7,8
    # Block (1,0): rows 2,3 cols 0,1 -> global 10,11,15,16
    # Block (1,1): rows 2,3 cols 2,3 -> global 12,13,17,18
    
    # For the 8th term (A22 * B22), we operate on indices 1..4 (0-indexed 1,2,3,4)
    # So local 0..3 map to global 1..4.
    # Local (0,0) -> Global (1,1), (1,2), (2,1), (2,2) -> Indices 6,7,11,12?
    # Global index for A[i,j] is i*5+j.
    # A22 is A[1:5, 1:5].
    # Local i, j in 0..3 -> Global i+1, j+1.
    # U index: (i+1)*5 + (j+1).
    # V index: (j+1)*5 + (k+1).
    # W index: (k+1)*5 + (i+1).
    
    # Strassen 4x4: 7 outer multiplications
    # M1: A11+A22, B11+B22
    # M2: A21+A22, B11
    # M3: A11, B12-B22
    # M4: A22, B21-B11
    # M5: A11+A12, B22
    # M6: A21-A11, B11+B12
    # M7: A12-A22, B21+B22
    
    # We define the linear combinations for A and B for each M_k
    # A_comb[k] is a list of (coeff, local_i, local_j)
    # B_comb[k] is a list of (coeff, local_j, local_k)
    
    # Local indices 0,1,2,3 correspond to global 1,2,3,4
    # Blocks in local 4x4:
    # A11: (0,0),(0,1),(1,0),(1,1)
    # A12: (0,2),(0,3),(1,2),(1,3)
    # A21: (2,0),(2,1),(3,0),(3,1)
    # A22: (2,2),(2,3),(3,2),(3,3)
    
    A_11 = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    A_12 = [(1, 0, 2), (1, 0, 3), (1, 1, 2), (1, 1, 3)]
    A_21 = [(1, 2, 0), (1, 2, 1), (1, 3, 0), (1, 3, 1)]
    A_22 = [(1, 2, 2), (1, 2, 3), (1, 3, 2), (1, 3, 3)]
    
    B_11 = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    B_12 = [(1, 0, 2), (1, 0, 3), (1, 1, 2), (1, 1, 3)]
    B_21 = [(1, 2, 0), (1, 2, 1), (1, 3, 0), (1, 3, 1)]
    B_22 = [(1, 2, 2), (1, 2, 3), (1, 3, 2), (1, 3, 3)]
    
    def add_lists(l1, l2, coeff=1):
        res = list(l1)
        for c, i, j in l2:
            res.append((c*coeff, i, j))
        return res
    
    def sub_lists(l1, l2):
        res = list(l1)
        for c, i, j in l2:
            res.append((-c, i, j))
        return res
    
    A_M1 = add_lists(A_11, A_22)
    B_M1 = add_lists(B_11, B_22)
    
    A_M2 = add_lists(A_21, A_22)
    B_M2 = B_11
    
    A_M3 = A_11
    B_M3 = sub_lists(B_12, B_22)
    
    A_M4 = A_22
    B_M4 = sub_lists(B_21, B_11)
    
    A_M5 = add_lists(A_11, A_12)
    B_M5 = B_22
    
    A_M6 = sub_lists(A_21, A_11)
    B_M6 = add_lists(B_11, B_12)
    
    A_M7 = sub_lists(A_12, A_22)
    B_M7 = add_lists(B_21, B_22)
    
    combos = [
        (A_M1, B_M1),
        (A_M2, B_M2),
        (A_M3, B_M3),
        (A_M4, B_M4),
        (A_M5, B_M5),
        (A_M6, B_M6),
        (A_M7, B_M7)
    ]
    
    for k, (A_comb, B_comb) in enumerate(combos):
        # A_comb and B_comb are lists of (coeff, i, j)
        # We need to form U_block, V_block, W_block for this M_k
        # by applying the inner Strassen (U2, V2, W2)
        
        # U_block[i_global, col_offset + r]
        # V_block[j_global, col_offset + r]
        # W_block[k_global, col_offset + r]
        
        # For each term r in 0..6 of inner Strassen:
        for r in range(7):
            col_idx = col + r
            
            # Compute U contribution
            # U2[i_local, r] is coeff for A_local[i_local]
            # A_local[i_local] is a linear combo of A_global
            # A_local[i] = sum_{(c, ii, jj) in A_comb, ii==i} c * A_global[ii, jj]
            # So U_global[ii, jj] += U2[i_local, r] * c
            
            for i_local in range(4):
                u_val = U2[i_local][r]
                if u_val != 0:
                    for (c, ii, jj) in A_comb:
                        if ii == i_local:
                            global_i = ii + 1
                            global_j = jj + 1
                            U[global_i * 5 + global_j][col_idx] += u_val * c
                            
            # Compute V contribution
            for j_local in range(4):
                v_val = V2[j_local][r]
                if v_val != 0:
                    for (c, jj, kk) in B_comb:
                        if jj == j_local:
                            global_j = jj + 1
                            global_k = kk + 1
                            V[global_j * 5 + global_k][col_idx] += v_val * c
                            
            # Compute W contribution
            # W2[i_local, r] is coeff for C_local[i_local]
            # C_local[i_local] contributes to C_global
            # C_local is 2x2.
            # C_local[0,0] -> C_global[0,0] (top-left of 4x4) AND C_global[2,2] (bottom-right)
            # C_local[0,1] -> C_global[0,1] AND C_global[2,3]
            # C_local[1,0] -> C_global[1,0] AND C_global[3,2]
            # C_local[1,1] -> C_global[1,1] AND C_global[3,3]
            
            for i_local in range(4):
                w_val = W2[i_local][r]
                if w_val != 0:
                    if i_local == 0: # 0,0
                        # C_global[0,0] and C_global[2,2]
                        # Global indices: 1,1 -> 6; 3,3 -> 18
                        W[1 * 5 + 1][col_idx] += w_val
                        W[3 * 5 + 3][col_idx] += w_val
                    elif i_local == 1: # 0,1
                        # C_global[0,1] and C_global[2,3]
                        # Global indices: 1,2 -> 7; 3,4 -> 19
                        W[1 * 5 + 2][col_idx] += w_val
                        W[3 * 5 + 4][col_idx] += w_val
                    elif i_local == 2: # 1,0
                        # C_global[1,0] and C_global[3,2]
                        # Global indices: 2,1 -> 11; 4,3 -> 23
                        W[2 * 5 + 1][col_idx] += w_val
                        W[4 * 5 + 3][col_idx] += w_val
                    elif i_local == 3: # 1,1
                        # C_global[1,1] and C_global[3,3]
                        # Global indices: 2,2 -> 12; 4,4 -> 24
                        W[2 * 5 + 2][col_idx] += w_val
                        W[4 * 5 + 4][col_idx] += w_val
                            
        col += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
