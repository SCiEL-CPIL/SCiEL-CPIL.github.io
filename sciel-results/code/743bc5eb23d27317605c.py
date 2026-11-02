"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Schoolbook as fallback
    rank_school = n * m * p  # 96
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Strategy: Split 4x4x6 into 4x4x4 + 4x4x2
    # Use Strassen-based rank 49 for 4x4x4 part
    # Use schoolbook rank 32 for 4x4x2 part
    # Total rank = 49 + 32 = 81
    
    # First, build the 4x4x4 Strassen decomposition (rank 49)
    # This is the Laderman decomposition for 4x4x4
    
    # Laderman's algorithm for 4x4x4 matrix multiplication (rank 49)
    # We'll construct it using block Strassen with 2x2 blocks
    
    # Actually, let me use the known rank 49 decomposition for 4x4x4
    # The standard approach: split 4x4 into four 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each Ci is a sum of two 2x2x2 products.
    # Using Strassen for each pair:
    # For C11: use Strassen on (A11,A12) and (B11,B21) treating them as 4x4 blocks? No.
    
    # Better: use the known 49-rank decomposition directly.
    # Let me construct it via nested Strassen.
    
    # Approach: 4x4x4 = two 2x2x4 multiplications + two 2x4x2 multiplications? No.
    
    # Let me use the standard construction:
    # Split into 2x2 blocks. We have 4 output 2x2 blocks, each requiring 
    # 2 scalar 2x2x2 products. Total 8 2x2x2 products.
    # If we use Strassen (rank 7) for each: 8*7 = 56. Still worse than 49.
    
    # The rank 49 comes from a more clever sharing. Let me look up Laderman.
    
    # Laderman 1976: 4x4x4 with rank 49.
    # The construction uses 7 Strassen-like steps but with more sharing.
    
    # Actually, let me just use a known explicit 49-term decomposition.
    # Or better: use the fact that we can do 4x4x4 in rank 49 via the following:
    
    # I'll construct it using the "block Strassen" approach where we treat
    # the 4x4 as 2x2 blocks of 2x2, and use a 2x2x2 Strassen on the block level,
    # but with the inner 2x2 multiplications done in a shared way.
    
    # For simplicity and correctness, let me use a different approach:
    # Use the known that 4x4x4 can be done in rank 49.
    # I'll hardcode the 49 rank-1 terms.
    
    # Actually, let me construct it properly. The Laderman decomposition:
    
    # Let me define the 49 terms for 4x4x4.
    # I'll use the following construction:
    # 
    # The 4x4x4 tensor can be decomposed as follows (using 2x2 block structure):
    # 
    # Define the following intermediate products (each is a 2x2x2 Strassen, rank 7):
    # 
    # Actually, the cleanest way: use the "Kronecker product" approach.
    # 4x4x4 = (2x2x2) ⊗ (2x2x2) would give rank 7*7=49.
    # 
    # Yes! The Kronecker product of two Strassen decompositions gives rank 49 for 4x4x4.
    # 
    # If S = (U_S, V_S, W_S) is a rank-7 decomposition of 2x2x2,
    # then (U_S ⊗ U_S, V_S ⊗ V_S, W_S ⊗ W_S) is a rank-49 decomposition of 4x4x4.
    
    # Let's construct this!
    
    # Strassen 2x2x2 decomposition (rank 7):
    # U_S has shape (4, 7), V_S has shape (4, 7), W_S has shape (4, 7)
    # where indices are (row, col) flattened.
    
    # For 2x2x2: A is 2x2, B is 2x2, C is 2x2
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # W index: k*2+i (col of B = row of C is k, row of A = col of C is i)
    
    # Strassen terms:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # U_S[i*2+j, r] = coefficient of A[i,j] in term r
    # V_S[j*2+k, r] = coefficient of B[j,k] in term r
    # W_S[k*2+i, r] = coefficient of C[k,i] in term r... wait
    
    # Let me be careful with indices.
    # A is 2x2: A[i,j] where i=row, j=col
    # B is 2x2: B[j,k] where j=row, k=col
    # C is 2x2: C[i,k] where i=row, k=col
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # U index: i*2+j (row*cols + col of A)
    # V index: j*2+k (row*cols + col of B)
    # W index: k*2+i (col of B = row of C, row of A = col of C) -> W[k*2+i]
    
    # Strassen:
    # P1: A coefs: A00=1, A11=1; B coefs: B00=1, B11=1; C coefs: C00=1, C11=1
    # P2: A coefs: A10=1, A11=1; B coefs: B00=1; C coefs: C10=1
    # P3: A coefs: A00=1; B coefs: B10=1, B11=-1; C coefs: C01=1, C11=1
    # P4: A coefs: A11=1; B coefs: B01=1, B00=-1; C coefs: C00=1, C10=1
    # P5: A coefs: A00=1, A01=1; B coefs: B11=1; C coefs: C00=-1, C01=1
    # P6: A coefs: A10=1, A00=-1; B coefs: B00=1, B01=1; C coefs: C11=1
    # P7: A coefs: A01=1, A11=-1; B coefs: B10=1, B11=1; C coefs: C00=1, C11=-1
    
    U_S = [[0]*7 for _ in range(4)]
    V_S = [[0]*7 for _ in range(4)]
    W_S = [[0]*7 for _ in range(4)]
    
    # Term 0 (P1):
    U_S[0][0] = 1; U_S[3][0] = 1  # A00, A11
    V_S[0][0] = 1; V_S[3][0] = 1  # B00, B11
    W_S[0][0] = 1; W_S[3][0] = 1  # C00 (k=0,i=0), C11 (k=1,i=1)
    
    # Term 1 (P2):
    U_S[2][1] = 1; U_S[3][1] = 1  # A10, A11
    V_S[0][1] = 1  # B00
    W_S[2][1] = 1  # C10 (k=0,i=1)
    
    # Term 2 (P3):
    U_S[0][2] = 1  # A00
    V_S[2][2] = 1; V_S[3][2] = -1  # B10, B11
    W_S[1][2] = 1; W_S[3][2] = 1  # C01 (k=1,i=0), C11 (k=1,i=1)
    
    # Term 3 (P4):
    U_S[3][3] = 1  # A11
    V_S[1][3] = 1; V_S[0][3] = -1  # B01, B00
    W_S[0][3] = 1; W_S[2][3] = 1  # C00 (k=0,i=0), C10 (k=0,i=1)
    
    # Term 4 (P5):
    U_S[0][4] = 1; U_S[1][4] = 1  # A00, A01
    V_S[3][4] = 1  # B11
    W_S[0][4] = -1; W_S[1][4] = 1  # C00 (k=0,i=0)=-1, C01 (k=1,i=0)=1
    
    # Term 5 (P6):
    U_S[2][5] = 1; U_S[0][5] = -1  # A10, A00
    V_S[0][5] = 1; V_S[1][5] = 1  # B00, B01
    W_S[3][5] = 1  # C11 (k=1,i=1)
    
    # Term 6 (P7):
    U_S[1][6] = 1; U_S[3][6] = -1  # A01, A11
    V_S[2][6] = 1; V_S[3][6] = 1  # B10, B11
    W_S[0][6] = 1; W_S[3][6] = -1  # C00 (k=0,i=0)=1, C11 (k=1,i=1)=-1
    
    # Now construct 4x4x4 via Kronecker product.
    # A is 4x4, B is 4x4, C is 4x4.
    # A[i,j] where i,j in 0-3
    # B[j,k] where j,k in 0-3
    # C[i,k] where i,k in 0-3
    # W index: k*4+i
    
    # The Kronecker product:
    # U_4[i*4+j, r1*7+r2] = U_S[i1*2+j1, r1] * U_S[i2*2+j2, r2]
    # where i = i1*2+i2, j = j1*2+j2
    # V_4[j*4+k, r1*7+r2] = V_S[j1*2+k1, r1] * V_S[j2*2+k2, r2]
    # where j = j1*2+j2, k = k1*2+k2
    # W_4[k*4+i, r1*7+r2] = W_S[k1*2+i1, r1] * W_S[k2*2+i2, r2]
    # where k = k1*2+k2, i = i1*2+i2
    
    rank_444 = 49
    U_444 = [[0]*rank_444 for _ in range(16)]
    V_444 = [[0]*rank_444 for _ in range(16)]
    W_444 = [[0]*rank_444 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1*7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1*2 + i2
                            j = j1*2 + j2
                            val_u = U_S[i1*2+j1][r1] * U_S[i2*2+j2][r2]
                            if val_u != 0:
                                U_444[i*4+j][r] += val_u
                            
                            val_v = V_S[j1*2+j2][r1] * V_S[i1*2+i2][r2]
                            # Wait, this isn't right. Let me reconsider.
                            # 
                            # The Kronecker product for matrix multiplication:
                            # If we have 2x2x2 with factors (U_S, V_S, W_S),
                            # then for 4x4x4, we decompose each 4x4 matrix into
                            # 2x2 blocks of 2x2.
                            # 
                            # A_4x4 = [[A11, A12], [A21, A22]] where each Aij is 2x2
                            # B_4x4 = [[B11, B12], [B21, B22]]
                            # C_4x4 = [[C11, C12], [C21, C22]]
                            # 
                            # C11 = A11*B11 + A12*B21
                            # C12 = A11*B12 + A12*B22
                            # C21 = A21*B11 + A22*B21
                            # C22 = A21*B12 + A22*B22
                            #
                            # This is NOT a simple Kronecker product. The Kronecker
                            # product would give C = A ⊗ B which is different.
                            #
                            # Let me reconsider. The correct approach for 4x4x4 rank 49:
                            #
                            # Actually, I recall that the rank of 4x4x4 is at most 49,
                            # and this is achieved by a specific construction, not simply
                            # the Kronecker product of two Strassen.
                            #
                            # Let me use a different approach: just use the block method
                            # where we have 8 2x2x2 products and use Strassen for each,
                            # giving rank 56. Then combine with the 4x4x2 part.
                            #
                            # 56 + 32 = 88. Still better than 96.
                            #
                            # Or, let me try to get the actual rank 49.
                            #
                            # For now, let me use the 56-rank 4x4x4 (8 Strassen terms)
                            # and 32-rank 4x4x2 (schoolbook), total 88.
                            pass
    
    # Let me restart with a cleaner approach.
    # Use 8 2x2x2 Strassen products for 4x4x4 (rank 56)
    # Use schoolbook for 4x4x2 (rank 32)
    # Total: 88
    
    # 4x4x4 part: 8 2x2x2 products
    # A is 4x4, split into 4 blocks of 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
    # B is 4x4, split into 4 blocks: B[0,0], B[0,1], B[1,0], B[1,1]
    # C is 4x4, split into 4 blocks: C[0,0], C[0,1], C[1,0], C[1,1]
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # Each Ci is a sum of two 2x2x2 products.
    # Total 8 2x2x2 products, each with Strassen rank 7 -> 56.
    
    rank_444 = 56
    U_444 = [[0]*rank_444 for _ in range(16)]
    V_444 = [[0]*rank_444 for _ in range(16)]
    W_444 = [[0]*rank_444 for _ in range(16)]
    
    col = 0
    for a_row in range(2):  # row block of A/C
        for a_col in range(2):  # col block of A / row block of B
            for b_col in range(2):  # col block of B/C
                # Product: A[a_row, a_col] * B[a_col, b_col] contributes to C[a_row, b_col]
                for (a_coefs, b_coefs, c_coefs) in [
                    ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                    ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
                    ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
                    ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
                    ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
                    ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
                    ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
                ]:
                    for (ar, ac), coef in a_coefs.items():
                        gi = 2*a_row + ar
                        gj = 2*a_col + ac
                        U_444[gi*4 + gj][col] += coef
                    for (br, bc), coef in b_coefs.items():
                        gi = 2*a_col + br
                        gj = 2*b_col + bc
                        V_444[gi*4 + gj][col] += coef
                    for (cr, cc), coef in c_coefs.items():
                        # C[a_row, b_col] block, C[k, i] where k=col of B, i=row of A
                        gi = 2*b_col + cc  # k
                        gj = 2*a_row + cr  # i
                        W_444[gi*4 + gj][col] += coef
                    col += 1
    
    assert col == rank_444, f"444 col={col}, expected {rank_444}"
    
    # 4x4x2 part: schoolbook, rank 32
    rank_442 = 32
    U_442 = [[0]*rank_442 for _ in range(16)]
    V_442 = [[0]*rank_442 for _ in range(8)]  # 4x2
    W_442 = [[0]*rank_442 for _ in range(8)]  # 2x4
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_442[i*4 + j][col] = 1
                V_442[j*2 + k][col] = 1
                W_442[k*4 + i][col] = 1
                col += 1
    
    assert col == rank_442, f"442 col={col}, expected {rank_442}"
    
    # Now combine for 4x4x6:
    # A is 4x4, B is 4x6, C is 4x6
    # Split B into B_left (4x4) and B_right (4x2)
    # C = A * B_left + A * B_right
    # C_left (4x4) = A * B_left -> use 4x4x4 decomposition
    # C_right (4x2) = A * B_right -> use 4x4x2 decomposition
    
    total_rank = rank_444 + rank_442  # 56 + 32 = 88
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(24)]  # 4x6
    W = [[0]*total_rank for _ in range(24)]  # 6x4
    
    # 4x4x4 part: columns 0-55
    for i in range(16):
        for r in range(rank_444):
            U[i][r] = U_444[i][r]
    for j in range(4):
        for k in range(4):
            for r in range(rank_444):
                V[j*6 + k][r] = V_444[j*4 + k][r]
    for k in range(4):
        for i in range(4):
            for r in range(rank_444):
                W[k*4 + i][r] = W_444[k*4 + i][r]
    
    # 4x4x2 part: columns 56-87
    offset = rank_444
    for i in range(16):
        for r in range(rank_442):
            U[i][offset + r] = U_442[i][r]
    for j in range(4):
        for k in range(2):
            for r in range(rank_442):
                V[j*6 + (4 + k)][offset + r] = V_442[j*2 + k][r]
    for k in range(2):
        for i in range(4):
            for r in range(rank_442):
                W[(4 + k)*4 + i][offset + r] = W_442[k*4 + i][r]
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(total_rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        best_rank = total_rank
        best_U = U
        best_V = V
        best_W = W
    
    # Check deadline
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
