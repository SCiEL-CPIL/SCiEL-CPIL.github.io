import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    rank_school = n * m * p  # 42
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # --- Known Hybrid Rank 39 Construction ---
    # Decomposition: 2x3x7 -> (2x2x2) + (2x1x2) + (2x1x2) + (2x1x1)
    # Rank 7 (Strassen) + 4 + 4 + 2 = 17 per block? 
    # Wait, Strassen is for 2x2x2. 
    # Splitting A (2x3) into [A1 (2x2), A2 (2x1)].
    # Splitting B (3x7) into [B1 (2x7), B2 (1x7)].
    # C = A1*B1 + A2*B2.
    
    # A1*B1: 2x2x2 tensor. Rank 7 (Strassen).
    # A2*B2: 2x1x2 tensor. Rank 4.
    # Total Rank = 7 + 4 = 11? 
    # Wait, B1 is 2x7. It's not 2x2.
    # The split must be in the middle dimension m.
    # m=3. Split into 2 and 1.
    # A (2x3) = [A_left (2x2), A_right (2x1)]
    # B (3x7) = [B_top (2x7); B_bottom (1x7)]
    # C = A_left @ B_top + A_right @ B_bottom
    
    # Term 1: A_left (2x2) @ B_top (2x7)
    # This is a 2x2x7 multiplication.
    # We can decompose B_top into 3 blocks of 2 columns + 1 column?
    # Or just treat 2x2x2 (for first 2 cols of B) and 2x2x1 (for rest).
    
    # Let's try to build Rank 39 explicitly as hinted.
    # Hint says: "hybrid Strassen/Schoolbook ... splitting inner dimension ... rank 39".
    # 42 - 39 = 3 saved.
    # Maybe one 2x2x2 block uses Strassen (7) instead of Schoolbook (8).
    # If we can identify a 2x2x2 sub-problem, we save 1.
    # 3 saved means 3 such sub-problems?
    # But we have 7 columns in B.
    # C = A @ B.
    # We can view C as [C_0, C_1, ..., C_6] where C_k is 2x1 vector.
    # C_k = A @ B[:, k].
    # B[:, k] is 3x1.
    # This is 2x3x1 multiplication. Rank 6? (2x3 + 3x1 = 5? No, rank is 6).
    # Schoolbook for 2x3x1 is 6.
    
    # Let's look at 2x2x2.
    # If we group columns of B into pairs?
    # B is 3x7.
    # Split B into B1 (3x2), B2 (3x2), B3 (3x2), B4 (3x1).
    # C = A B1 + A B2 + A B3 + A B4.
    # A B1 is 2x2. (2x3 @ 3x2).
    # Rank of 2x3x2?
    # Schoolbook is 6*2 = 12.
    # If we use Strassen-like for the 2x2 part of the output?
    
    # Actually, let's just try to construct the Rank 39 solution found by the "island 9" program.
    # The prompt says: "reducing the rank to 39".
    # Mechanism: "splitting the inner dimension and using Strassen for 2x2x2 blocks".
    
    # Let's try to manually construct a rank 39 solution.
    # Total terms: 39.
    # Schoolbook: 42.
    # Difference: 3.
    # This implies 3 "Strassen savings".
    # Each Strassen on 2x2x2 saves 1.
    # So we need to identify three independent 2x2x2 sub-problems.
    
    # A (2x3). B (3x7).
    # Let's partition the inner dimension j=0,1,2.
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k].
    
    # If we can combine some of these products.
    # Consider the sub-matrix of C for rows i=0,1 and columns k=0,1.
    # This is a 2x2 output block.
    # It depends on A (2x3) and B (3x2).
    # This is a 2x3x2 tensor.
    # Can we decompose 2x3x2 with rank < 12?
    # 2x3x2 is equivalent to 2x2x2 + 2x1x2.
    # Rank(2x2x2) = 7.
    # Rank(2x1x2) = 4.
    # Total Rank = 11.
    # Schoolbook for 2x3x2 is 12.
    # So we save 1.
    
    # We have 7 columns in B.
    # We can form floor(7/2) = 3 blocks of 2 columns, and 1 block of 1 column.
    # Blocks:
    # 1. B[:, 0:2] (3x2). Output C[:, 0:2] (2x2).
    # 2. B[:, 2:4] (3x2). Output C[:, 2:4] (2x2).
    # 3. B[:, 4:6] (3x2). Output C[:, 4:6] (2x2).
    # 4. B[:, 6:7] (3x1). Output C[:, 6:7] (2x1).
    
    # Rank for Block 1 (2x3x2): 11.
    # Rank for Block 2 (2x3x2): 11.
    # Rank for Block 3 (2x3x2): 11.
    # Rank for Block 4 (2x3x1): 6.
    
    # Total Rank = 11 + 11 + 11 + 6 = 39.
    # This matches the hint!
    
    # Now, let's construct the matrices U, V, W for this decomposition.
    
    # We will build U, V, W as horizontal concatenations of the factors for each block.
    
    # --- Block Construction ---
    
    # Helper: Get Strassen factors for 2x2x2
    # Returns U_s (4, 7), V_s (4, 7), W_s (4, 7)
    # Indices: U[i,j], V[j,k], W[k,i]
    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Standard Strassen decomposition
        # m1 = (a11 - a22) * (b22 + b11)
        # m2 = (a21 + a22) * b11
        # m3 = a11 * (b12 - b22)
        # m4 = a22 * (b21 - b11)
        # m5 = (a11 + a22) * (b11 + b22)
        # m6 = (a11 - a12) * (b12 + b22)
        # m7 = (a21 + a12) * (b21 - b11)
        
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m5
        # c22 = m1 - m2 + m3 + m4
        
        # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # Term 0: m1
        U[0, 0] = 1; U[3, 0] = -1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 1: m2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # Term 2: m3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # Term 3: m4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[3, 3] = 1
        
        # Term 4: m5
        U[0, 4] = 1; U[3, 4] = 1
        V[0, 4] = 1; V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1; W[2, 4] = 1; W[3, 4] = 1
        
        # Term 5: m6
        U[0, 5] = 1; U[1, 5] = -1
        V[1, 5] = 1; V[3, 5] = 1
        W[1, 5] = 1
        
        # Term 6: m7
        U[2, 6] = 1; U[1, 6] = 1
        V[2, 6] = 1; V[0, 6] = -1
        W[0, 6] = 1; W[2, 6] = 1
        
        return U, V, W

    # Helper: Get Schoolbook factors for 2x1x2 (A is 2x1, B is 1x2)
    # Wait, in our block decomposition, the "inner" dim is always 3.
    # The block is 2x3x2.
    # We decomposed 2x3x2 into 2x2x2 (using rows 0,1 of A and rows 0,1 of B) + 2x1x2 (using row 2 of A and rows 0,1 of B).
    # Wait, A is 2x3. B is 3x2.
    # A = [[A00, A01], [A10, A11], [A20, A21]] ? No, A is 2x3.
    # A = [[a00, a01, a02], [a10, a11, a12]].
    # B = [[b00, b01], [b10, b11], [b20, b21]].
    # C = A @ B.
    # C = A[0:2, 0:2] @ B[0:2, 0:2] + A[0:2, 2:3] @ B[2:3, 0:2].
    # First term: 2x2x2.
    # Second term: 2x1x2.
    
    def get_factors_2x1x2():
        # A is 2x1 (indices 0,1). B is 1x2 (indices 0,1).
        # C is 2x2 (indices 0,1 for rows, 0,1 for cols).
        # Rank 4.
        # U (2, 4), V (2, 4), W (4, 4)?
        # No, U shape (n*m, rank). Here n=2, m=1. U (2, 4).
        # V shape (m*p, rank). Here m=1, p=2. V (2, 4).
        # W shape (p*n, rank). Here p=2, n=2. W (4, 4).
        
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        # Term 0: a0 * b0 -> c00
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1  # c00 index is 0*2+0 = 0
        
        # Term 1: a0 * b1 -> c01
        U[0, 1] = 1
        V[1, 1] = 1
        W[1, 1] = 1  # c01 index is 1*2+0 = 2? 
        # Wait, W index convention: W[k*n+i, r].
        # p=2, n=2.
        # k=0, i=0 -> 0.
        # k=1, i=0 -> 2.
        # k=0, i=1 -> 1.
        # k=1, i=1 -> 3.
        
        # Let's re-verify W index: W[k*n+i, r].
        # Term 0: a0 (U[0]) * b0 (V[0]) -> c00 (W[0]). Correct.
        # Term 1: a0 (U[0]) * b1 (V[1]) -> c01 (W[2]).
        W[2, 1] = 1
        
        # Term 2: a1 (U[1]) * b0 (V[0]) -> c10 (W[1]).
        U[1, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # Term 3: a1 (U[1]) * b1 (V[1]) -> c11 (W[3]).
        U[1, 3] = 1
        V[1, 3] = 1
        W[3, 3] = 1
        
        return U, V, W

    def get_factors_2x1x1():
        # A is 2x1. B is 1x1. C is 2x1.
        # Rank 2.
        U = np.zeros((2, 2))
        V = np.zeros((1, 2))
        W = np.zeros((2, 2))
        
        # W index: k*n+i. p=1, n=2.
        # k=0. i=0 -> 0. i=1 -> 1.
        
        # Term 0: a0 * b0 -> c0 (W[0])
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Term 1: a1 * b0 -> c1 (W[1])
        U[1, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        return U, V, W

    # --- Assembling the Full Decomposition ---
    
    # Total Rank = 39.
    # U shape: (6, 39). (n*m = 2*3 = 6).
    # V shape: (21, 39). (m*p = 3*7 = 21).
    # W shape: (14, 39). (p*n = 7*2 = 14).
    
    U_total = np.zeros((6, 39))
    V_total = np.zeros((21, 39))
    W_total = np.zeros((14, 39))
    
    col_offset = 0
    
    # Define the blocks
    # Block 1: k=0,1
    # Block 2: k=2,3
    # Block 3: k=4,5
    # Block 4: k=6
    
    blocks = [
        {"cols": [0, 1], "type": "2x3x2"},
        {"cols": [2, 3], "type": "2x3x2"},
        {"cols": [4, 5], "type": "2x3x2"},
        {"cols": [6],    "type": "2x3x1"}
    ]
    
    for block in blocks:
        cols = block["cols"]
        btype = block["type"]
        
        if btype == "2x3x2":
            # Decompose into 2x2x2 (j=0,1) and 2x1x2 (j=2)
            
            # Part 1: 2x2x2
            # A part: rows 0,1 of A, cols 0,1.
            # B part: rows 0,1 of B, cols k1, k2.
            # C part: rows 0,1 of C, cols k1, k2.
            
            U_s, V_s, W_s = get_strassen_2x2x2()
            
            # Map to global indices
            # U global: [i, j]. i=0,1. j=0,1.
            # V global: [j, k]. j=0,1. k=cols[0], cols[1].
            # W global: [k, i]. k=cols[0], cols[1]. i=0,1.
            
            for r in range(7):
                # U
                for u_idx in range(4):
                    i = u_idx // 2
                    j = u_idx % 2
                    val = U_s[u_idx, r]
                    if val != 0:
                        U_total[i * 3 + j, col_offset + r] += val
                        
                # V
                for v_idx in range(4):
                    j = v_idx // 2
                    k_local = v_idx % 2
                    k_global = cols[k_local]
                    val = V_s[v_idx, r]
                    if val != 0:
                        V_total[j * 7 + k_global, col_offset + r] += val
                        
                # W
                for w_idx in range(4):
                    k_local = w_idx // 2
                    i = w_idx % 2
                    k_global = cols[k_local]
                    val = W_s[w_idx, r]
                    if val != 0:
                        W_total[k_global * 2 + i, col_offset + r] += val
                        
            col_offset += 7
            
            # Part 2: 2x1x2
            # A part: rows 0,1 of A, col 2.
            # B part: row 2 of B, cols k1, k2.
            # C part: rows 0,1 of C, cols k1, k2.
            
            U_212, V_212, W_212 = get_factors_2x1x2()
            
            for r in range(4):
                # U
                # U_212 indices: [i]. i=0,1.
                for u_idx in range(2):
                    i = u_idx
                    j = 2  # Fixed column 2 of A
                    val = U_212[u_idx, r]
                    if val != 0:
                        U_total[i * 3 + j, col_offset + r] += val
                        
                # V
                # V_212 indices: [k_local]. k_local=0,1.
                for v_idx in range(2):
                    j = 2  # Fixed row 2 of B
                    k_local = v_idx
                    k_global = cols[k_local]
                    val = V_212[v_idx, r]
                    if val != 0:
                        V_total[j * 7 + k_global, col_offset + r] += val
                        
                # W
                # W_212 indices: [k_local * 2 + i].
                for w_idx in range(4):
                    k_local = w_idx // 2
                    i = w_idx % 2
                    k_global = cols[k_local]
                    val = W_212[w_idx, r]
                    if val != 0:
                        W_total[k_global * 2 + i, col_offset + r] += val
                        
            col_offset += 4
            
        elif btype == "2x3x1":
            # Decompose into 2x2x1 (j=0,1) and 2x1x1 (j=2)
            # Wait, 2x2x1 rank is 4?
            # A (2x2) B (2x1).
            # C (2x1).
            # c0 = a00 b0 + a01 b1
            # c1 = a10 b0 + a11 b1
            # Rank 4.
            
            # Part 1: 2x2x1
            # U (4, 4), V (2, 4), W (2, 4)?
            # No, U (n*m, r) = (4, 4). V (m*p, r) = (2, 4). W (p*n, r) = (2, 4).
            
            # Let's just use schoolbook for 2x2x1 and 2x1x1.
            # 2x2x1:
            # Term 0: a00 b0 -> c0
            # Term 1: a01 b1 -> c0
            # Term 2: a10 b0 -> c1
            # Term 3: a11 b1 -> c1
            
            for r in range(4):
                if r == 0:
                    U_total[0, col_offset] = 1
                    V_total[cols[0], col_offset] = 1
                    W_total[cols[0] * 2 + 0, col_offset] = 1
                elif r == 1:
                    U_total[1, col_offset] = 1
                    V_total[cols[0] + 1, col_offset] = 1 # Wait, B row index?
                    # B is 2x1 in this subproblem. Rows 0,1 of B correspond to j=0,1.
                    # So V index is j*1 + k_local. j=0, k=0. V[0].
                    # j=1, k=0. V[1].
                    V_total[1 * 1 + 0, col_offset] = 1
                    # Correction: V shape is (m*p, r). Here m=2, p=1.
                    # V index = j*1 + k.
                    # For r=1: a01 (U[1]) * b1 (V[1]) -> c0 (W[0]).
                    U_total[1, col_offset + 1] = 1
                    V_total[1, col_offset + 1] = 1
                    W_total[cols[0] * 2 + 0, col_offset + 1] = 1
                elif r == 2:
                    U_total[2, col_offset + 2] = 1
                    V_total[0, col_offset + 2] = 1
                    W_total[cols[0] * 2 + 1, col_offset + 2] = 1
                elif r == 3:
                    U_total[3, col_offset + 3] = 1
                    V_total[1, col_offset + 3] = 1
                    W_total[cols[0] * 2 + 1, col_offset + 3] = 1
            
            col_offset += 4
            
            # Part 2: 2x1x1
            U_211, V_211, W_211 = get_factors_2x1x1()
            for r in range(2):
                for u_idx in range(2):
                    i = u_idx
                    j = 2
                    val = U_211[u_idx, r]
                    if val != 0:
                        U_total[i * 3 + j, col_offset + r] += val
                for v_idx in range(1):
                    j = 2
                    k_global = cols[0]
                    val = V_211[v_idx, r]
                    if val != 0:
                        V_total[j * 7 + k_global, col_offset + r] += val
                for w_idx in range(2):
                    i = w_idx
                    k_global = cols[0]
                    val = W_211[w_idx, r]
                    if val != 0:
                        W_total[k_global * 2 + i, col_offset + r] += val
            col_offset += 2

    # Verify Rank 39
    if col_offset == 39:
        # Quick verification with random matrices
        A = np.random.rand(2, 3)
        B = np.random.rand(3, 7)
        C_expected = A @ B
        
        # Compute C using decomposition
        C_calc = np.zeros((2, 7))
        # We need to "contract" U, V, W.
        # C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ??
        # No, the decomposition is sum_r U[:, r] V[:, r]^T W[:, r]^T?
        # Tensor T = sum_r u_r \otimes v_r \otimes w_r.
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r].
        # In our flattened notation:
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        # We want T[i, j, k] = 1 if it's a valid product term?
        # No, T represents the multiplication.
        # C[i, k] = sum_j T[i, j, k] A[i, j] B[j, k].
        
        # So we can check if the tensor T we constructed is the multiplication tensor.
        # T[i, j, k] should be 1 for all i, j, k?
        # No, T[i, j, k] = 1.
        
        # Let's verify T[i, j, k] == 1 for all i, j, k.
        valid = True
        for i in range(2):
            for j in range(3):
                for k in range(7):
                    u_idx = i * 3 + j
                    v_idx = j * 7 + k
                    w_idx = k * 2 + i
                    sum_val = np.dot(U_total[u_idx, :], V_total[v_idx, :] * W_total[w_idx, :])
                    if not np.isclose(sum_val, 1.0):
                        valid = False
                        # print(f"Error at {i},{j},{k}: {sum_val}")
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            best_result = {
                "U": U_total.tolist(),
                "V": V_total.tolist(),
                "W": W_total.tolist()
            }
            best_rank = 39
    else:
        print(f"Rank mismatch: {col_offset}")

    return best_result
