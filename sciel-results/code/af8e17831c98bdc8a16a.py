"""Program for Matrix multiplication 2,4,5 using Strassen-based block decomposition."""

def construct():
    # Dimensions: A (2x4), B (4x5), C (2x5)
    # Split B into B1 (4x4) and B2 (4x1)
    # C = A * B1 + A * B2
    # Use Strassen for A (2x4) * B1 (4x4) -> C1 (2x4)
    # Use Schoolbook for A (2x4) * B2 (4x1) -> C2 (2x1)
    
    rank_strassen = 7
    rank_schoolbook = 4
    total_rank = rank_strassen + rank_schoolbook
    
    # U has shape (n*m, rank) = (8, 11)
    # V has shape (m*p, rank) = (20, 11)
    # W has shape (p*n, rank) = (10, 11)
    
    U = [[0.0] * total_rank for _ in range(8)]
    V = [[0.0] * total_rank for _ in range(20)]
    W = [[0.0] * total_rank for _ in range(10)]
    
    # --- Strassen Part (indices 0 to 6) ---
    # A is 2x4. Split into A_left (2x2) and A_right (2x2)
    # A_left rows 0,1; A_right rows 2,3 (in flattened A: 0,1,2,3)
    # B1 is 4x4. Split into 2x2 blocks:
    #   B1_11 (rows 0,1; cols 0,1) -> indices 0,1,4,5 in B1
    #   B1_12 (rows 0,1; cols 2,3) -> indices 2,3,6,7 in B1
    #   B1_21 (rows 2,3; cols 0,1) -> indices 8,9,12,13 in B1
    #   B1_22 (rows 2,3; cols 2,3) -> indices 10,11,14,15 in B1
    # In global B (4x5), B1 columns are 0,1,2,3.
    # B indices: row j (0-3), col k (0-3) -> index j*5 + k
    # B1_11: j in {0,1}, k in {0,1} -> 0,1, 5,6
    # B1_12: j in {0,1}, k in {2,3} -> 2,3, 7,8
    # B1_21: j in {2,3}, k in {0,1} -> 10,11, 15,16
    # B1_22: j in {2,3}, k in {2,3} -> 12,13, 17,18
    
    # A indices: i (0-1), j (0-3) -> index i*4 + j
    # A_left: i in {0,1}, j in {0,1} -> 0,1, 4,5
    # A_right: i in {0,1}, j in {2,3} -> 2,3, 6,7
    
    # C1 is 2x4. Split into C1_left (2x2) and C1_right (2x2)
    # C1_left cols 0,1; C1_right cols 2,3
    # Global C indices: k (0-4), i (0-1) -> index k*2 + i
    # C1_left: k in {0,1}, i in {0,1} -> 0,1, 2,3
    # C1_right: k in {2,3}, i in {0,1} -> 4,5, 6,7
    
    # Strassen formulas for C_left and C_right:
    # C_left = [A_left*B1_11 + A_right*B1_21]
    # C_right = [A_left*B1_12 + A_right*B1_22]
    
    # We can apply Strassen to each 2x2x2 multiplication separately?
    # Or combine?
    # Actually, standard Strassen computes C = A*B for 2x2 matrices.
    # Here we have two separate multiplications:
    # 1. A_left (2x2) * B1_11 (2x2) + A_right (2x2) * B1_21 (2x2) = C_left (2x2)
    # 2. A_left (2x2) * B1_12 (2x2) + A_right (2x2) * B1_22 (2x2) = C_right (2x2)
    
    # This looks like two independent Strassen decompositions if we treat the sums carefully.
    # But wait, Strassen computes a product of two 2x2 matrices.
    # Here we have a sum of two products.
    # Let's use the 7-term decomposition for each product separately?
    # Product 1: A_left * B1_11 (7 terms)
    # Product 2: A_right * B1_21 (7 terms)
    # Sum them to get C_left.
    # Product 3: A_left * B1_12 (7 terms)
    # Product 4: A_right * B1_22 (7 terms)
    # Sum them to get C_right.
    # Total terms: 28.
    # Plus 4 schoolbook terms for the last column.
    # Total rank: 32.
    # Schoolbook is 40. 32 < 40. This is an improvement.
    
    # Can we do better?
    # Maybe combine the two left products?
    # C_left = A_left B1_11 + A_right B1_21
    # This is not a single matrix product.
    
    # Let's stick to 4 separate Strassen decompositions for the 4 blocks of B1.
    # Term indices 0-6: A_left * B1_11
    # Term indices 7-13: A_right * B1_21
    # Term indices 14-20: A_left * B1_12
    # Term indices 21-27: A_right * B1_22
    # Term indices 28-31: Schoolbook for column 4
    
    # Strassen M1..M7 for multiplying X (2x2) by Y (2x2):
    # M1 = (X11 + X22)(Y11 + Y22)
    # M2 = (X21 + X22)Y11
    # M3 = X11(Y12 - Y22)
    # M4 = X22(Y21 - Y11)
    # M5 = (X11 + X12)Y22
    # M6 = (X21 - X11)(Y11 + Y12)
    # M7 = (X12 - X22)(Y21 + Y22)
    
    # C11 = M1 - M3 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M5 - M6
    
    # Helper to add a Strassen block
    def add_strassen(offset_u, offset_v, offset_w, a_rows, b_rows, c_rows, start_idx):
        # a_rows: list of 4 global U indices corresponding to X11, X12, X21, X22
        # b_rows: list of 4 global V indices corresponding to Y11, Y12, Y21, Y22
        # c_rows: list of 4 global W indices corresponding to C11, C12, C21, C22
        
        # M1
        U[a_rows[0]][start_idx] += 1; U[a_rows[3]][start_idx] += 1
        V[b_rows[0]][start_idx] += 1; V[b_rows[3]][start_idx] += 1
        W[c_rows[0]][start_idx] += 1; W[c_rows[3]][start_idx] += 1
        
        # M2
        U[a_rows[2]][start_idx+1] += 1; U[a_rows[3]][start_idx+1] += 1
        V[b_rows[0]][start_idx+1] += 1
        W[c_rows[2]][start_idx+1] += 1; W[c_rows[3]][start_idx+1] -= 1
        
        # M3
        U[a_rows[0]][start_idx+2] += 1
        V[b_rows[1]][start_idx+2] += 1; V[b_rows[3]][start_idx+2] -= 1
        W[c_rows[0]][start_idx+2] -= 1; W[c_rows[1]][start_idx+2] += 1
        
        # M4
        U[a_rows[3]][start_idx+3] += 1
        V[b_rows[2]][start_idx+3] += 1; V[b_rows[0]][start_idx+3] -= 1
        W[c_rows[0]][start_idx+3] += 1; W[c_rows[2]][start_idx+3] += 1
        
        # M5
        U[a_rows[0]][start_idx+4] += 1; U[a_rows[1]][start_idx+4] += 1
        V[b_rows[3]][start_idx+4] += 1
        W[c_rows[0]][start_idx+4] -= 1; W[c_rows[1]][start_idx+4] += 1
        
        # M6
        U[a_rows[2]][start_idx+5] -= 1; U[a_rows[0]][start_idx+5] -= -1 # X21 - X11 => U21=1, U11=-1? No, X21 - X11. So U[X21]+=1, U[X11]+=-1.
        U[a_rows[2]][start_idx+5] = 1; U[a_rows[0]][start_idx+5] = -1
        V[b_rows[0]][start_idx+5] += 1; V[b_rows[1]][start_idx+5] += 1
        W[c_rows[3]][start_idx+5] -= 1
        
        # M7
        U[a_rows[1]][start_idx+6] -= 1; U[a_rows[3]][start_idx+6] -= -1 # X12 - X22 => U[X12]+=1, U[X22]+=-1
        U[a_rows[1]][start_idx+6] = 1; U[a_rows[3]][start_idx+6] = -1
        V[b_rows[2]][start_idx+6] += 1; V[b_rows[3]][start_idx+6] += 1
        W[c_rows[0]][start_idx+6] += 1; W[c_rows[3]][start_idx+6] += 1

    # Define indices
    # A_left: X11=A[0,0]=0, X12=A[0,1]=1, X21=A[1,0]=4, X22=A[1,1]=5
    # A_right: X11=A[0,2]=2, X12=A[0,3]=3, X21=A[1,2]=6, X22=A[1,3]=7
    
    # B1_11: Y11=B[0,0]=0, Y12=B[0,1]=1, Y21=B[1,0]=5, Y22=B[1,1]=6
    # B1_12: Y11=B[0,2]=2, Y12=B[0,3]=3, Y21=B[1,2]=7, Y22=B[1,3]=8
    # B1_21: Y11=B[2,0]=10, Y12=B[2,1]=11, Y21=B[3,0]=15, Y22=B[3,1]=16
    # B1_22: Y11=B[2,2]=12, Y12=B[2,3]=13, Y21=B[3,2]=17, Y22=B[3,3]=18
    
    # C_left: C11=C[0,0]=0, C12=C[1,0]=1, C21=C[0,1]=2, C22=C[1,1]=3
    # C_right: C11=C[2,0]=4, C12=C[3,0]=5, C21=C[2,1]=6, C22=C[3,1]=7
    
    # Block 1: A_left * B1_11 -> adds to C_left
    add_strassen(0, 0, 0, [0,1,4,5], [0,1,5,6], [0,1,2,3], 0)
    
    # Block 2: A_right * B1_21 -> adds to C_left
    add_strassen(0, 0, 0, [2,3,6,7], [10,11,15,16], [0,1,2,3], 7)
    
    # Block 3: A_left * B1_12 -> adds to C_right
    add_strassen(0, 0, 0, [0,1,4,5], [2,3,7,8], [4,5,6,7], 14)
    
    # Block 4: A_right * B1_22 -> adds to C_right
    add_strassen(0, 0, 0, [2,3,6,7], [12,13,17,18], [4,5,6,7], 21)
    
    # --- Schoolbook Part (indices 28 to 31) ---
    # Multiply A (2x4) by B2 (4x1) -> C2 (2x1)
    # B2 is column 4 of B.
    # B indices: j*5 + 4.
    # B[0,4]=4, B[1,4]=9, B[2,4]=14, B[3,4]=19
    # C2 indices: C[4,0]=8, C[4,1]=9
    
    # Term 28: A[0,0] * B[0,4] -> C[4,0]
    U[0][28] = 1; V[4][28] = 1; W[8][28] = 1
    # Term 29: A[0,1] * B[1,4] -> C[4,0]
    U[1][29] = 1; V[9][29] = 1; W[8][29] = 1
    # Term 30: A[0,2] * B[2,4] -> C[4,0]
    U[2][30] = 1; V[14][30] = 1; W[8][30] = 1
    # Term 31: A[0,3] * B[3,4] -> C[4,0]
    U[3][31] = 1; V[19][31] = 1; W[8][31] = 1
    
    # Wait, C is 2x5. C[i,k].
    # C[0,4] and C[1,4] are needed.
    # My schoolbook above only computes C[0,4] (i=0, k=4).
    # I need C[1,4] (i=1, k=4) as well.
    # Let's re-evaluate the schoolbook part.
    # A (2x4) * B2 (4x1).
    # Result is 2x1 vector.
    # C[0,4] = sum_j A[0,j] B[j,4]
    # C[1,4] = sum_j A[1,j] B[j,4]
    
    # So I need 4 terms for i=0 and 4 terms for i=1. Total 8 terms.
    # My previous count of 4 was for 2x2x1 (i.e. 2x1 * 1x1? No, 2x1 matrix times 1x1 vector is 2x1).
    # Wait, Schoolbook for n x m * m x p is n*m*p.
    # Here n=2, m=4, p=1. Rank = 2*4*1 = 8.
    # So total rank = 28 (Strassen) + 8 (Schoolbook) = 36.
    # 36 < 40. Still an improvement.
    
    # Let's extend the arrays to rank 36.
    # Re-initialize with rank 36
    U = [[0.0] * 36 for _ in range(8)]
    V = [[0.0] * 36 for _ in range(20)]
    W = [[0.0] * 36 for _ in range(10)]
    
    # Re-run Strassen parts (0-27)
    add_strassen(0, 0, 0, [0,1,4,5], [0,1,5,6], [0,1,2,3], 0)
    add_strassen(0, 0, 0, [2,3,6,7], [10,11,15,16], [0,1,2,3], 7)
    add_strassen(0, 0, 0, [0,1,4,5], [2,3,7,8], [4,5,6,7], 14)
    add_strassen(0, 0, 0, [2,3,6,7], [12,13,17,18], [4,5,6,7], 21)
    
    # Schoolbook for column 4 (indices 28-35)
    # i=0 terms
    U[0][28] = 1; V[4][28] = 1; W[8][28] = 1   # A[0,0]B[0,4] -> C[0,4]
    U[1][29] = 1; V[9][29] = 1; W[8][29] = 1   # A[0,1]B[1,4] -> C[0,4]
    U[2][30] = 1; V[14][30] = 1; W[8][30] = 1  # A[0,2]B[2,4] -> C[0,4]
    U[3][31] = 1; V[19][31] = 1; W[8][31] = 1  # A[0,3]B[3,4] -> C[0,4]
    
    # i=1 terms
    U[4][32] = 1; V[4][32] = 1; W[9][32] = 1   # A[1,0]B[0,4] -> C[1,4]
    U[5][33] = 1; V[9][33] = 1; W[9][33] = 1   # A[1,1]B[1,4] -> C[1,4]
    U[6][34] = 1; V[14][34] = 1; W[9][34] = 1  # A[1,2]B[2,4] -> C[1,4]
    U[7][35] = 1; V[19][35] = 1; W[9][35] = 1  # A[1,3]B[3,4] -> C[1,4]
    
    return {"U": U, "V": V, "W": W}
