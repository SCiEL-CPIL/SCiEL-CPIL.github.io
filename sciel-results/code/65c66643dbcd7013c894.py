"""Matrix multiplication 2x4x8 with Strassen-based rank reduction."""

# EVOLVE-BLOCK-START
def construct():
    # Dimensions
    n, m, p = 2, 4, 8
    
    # We decompose the 2x4x8 multiplication into two independent 2x4x4 multiplications:
    # C[:, 0:4] = A @ B[:, 0:4] and C[:, 4:8] = A @ B[:, 4:8]
    # Each 2x4x4 can be further decomposed into 4 2x2x2 Strassen multiplications (rank 7 each)
    # Total rank = 2 * 4 * 7 = 56
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For matrices A (2x2), B (2x2), C = A @ B (2x2):
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)(B11)
    # M3 = (A11)(B12 - B22)
    # M4 = (A22)(B21 - B11)
    # M5 = (A11 + A12)(B22)
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U entries for 2x2x2 Strassen (indices: i*2+j for A)
    # i is row of A (0,1), j is col of A (0,1)
    # A11 = A[0,0] = index 0
    # A12 = A[0,1] = index 1
    # A21 = A[1,0] = index 2
    # A22 = A[1,1] = index 3
    strassen_U = [
        [1, 0, 0, 1],  # M1: A11 + A22
        [0, 0, 1, 1],  # M2: A21 + A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11 + A12
        [-1, 0, 1, 0], # M6: A21 - A11
        [0, 1, 0, -1], # M7: A12 - A22
    ]
    
    # V entries for 2x2x2 Strassen (indices: j*2+k for B)
    # B11 = B[0,0] = index 0
    # B12 = B[0,1] = index 1
    # B21 = B[1,0] = index 2
    # B22 = B[1,1] = index 3
    strassen_V = [
        [1, 0, 0, 1],  # M1: B11 + B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1], # M3: B12 - B22
        [0, 0, 1, -1], # M4: B21 - B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11 + B12
        [0, 0, 1, 1],  # M7: B21 + B22
    ]
    
    # W entries for 2x2x2 Strassen (indices: k*2+i for C)
    # C11 = C[0,0] = index 0
    # C12 = C[0,1] = index 1
    # C21 = C[1,0] = index 2
    # C22 = C[1,1] = index 3
    strassen_W = [
        [1, 0, 0, 1],   # C11 = M1 + M4 - M5 + M7
        [0, 1, 1, 0],   # C12 = M3 + M5
        [0, 1, 1, 0],   # C21 = M2 + M4
        [1, -1, 1, 1],  # C22 = M1 - M2 + M3 + M6
    ]
    
    # Now build 2x4x4 decomposition:
    # A is 2x4, B is 4x4, C is 2x4
    # We can view this as 4 2x2x2 multiplications:
    # C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2]  (block (0,0))
    # C[:, 2:4] = A[:, 0:2] @ B[2:4, 0:2]  (block (1,0))
    # C[:, 0:2] += A[:, 2:4] @ B[0:2, 2:4]  (block (0,1))
    # C[:, 2:4] += A[:, 2:4] @ B[2:4, 2:4]  (block (1,1))
    # 
    # Wait, that's not quite right. Let me reconsider.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # For 2x4x4: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # We can split j into two groups: {0,1} and {2,3}
    # C[i, k] = sum_{j in {0,1}} A[i, j] B[j, k] + sum_{j in {2,3}} A[i, j] B[j, k]
    # The first term is a 2x2x4 multiplication, the second is also 2x2x4
    # 
    # Actually, let me think differently. Split both A's columns and B's rows into pairs:
    # Let A = [A_left | A_right] where A_left is 2x2 (cols 0,1) and A_right is 2x2 (cols 2,3)
    # Let B = [B_top; B_bottom] where B_top is 2x4 (rows 0,1) and B_bottom is 2x4 (rows 2,3)
    # Then C = A_left @ B_top + A_right @ B_bottom
    # Each term is 2x2 @ 2x4 = 2x4
    # A 2x2 @ 2x4 can be done as 2 2x2 @ 2x2 multiplications (split B's columns)
    # So total = 4 2x2 @ 2x2 multiplications, each rank 7 = 28 total
    #
    # For the full 2x4x8:
    # Split B's columns into two groups of 4: B_left (cols 0-3) and B_right (cols 4-7)
    # C_left = A @ B_left (2x4 @ 4x4 = 2x4), rank 28
    # C_right = A @ B_right (2x4 @ 4x4 = 2x4), rank 28
    # Total rank = 56
    
    # Build 2x4x4 decomposition (rank 28)
    # A is 2x4: indices i*4+j, i in {0,1}, j in {0,1,2,3}
    #   A[0,0]=idx0, A[0,1]=idx1, A[0,2]=idx2, A[0,3]=idx3
    #   A[1,0]=idx4, A[1,1]=idx5, A[1,2]=idx6, A[1,3]=idx7
    # B is 4x4: indices j*4+k, j in {0,1,2,3}, k in {0,1,2,3}
    #   B[0,0]=idx0, B[0,1]=idx1, B[0,2]=idx2, B[0,3]=idx3
    #   B[1,0]=idx4, B[1,1]=idx5, B[1,2]=idx6, B[1,3]=idx7
    #   B[2,0]=idx8, B[2,1]=idx9, B[2,2]=idx10, B[2,3]=idx11
    #   B[3,0]=idx12, B[3,1]=idx13, B[3,2]=idx14, B[3,3]=idx15
    # C is 2x4: indices k*2+i, i in {0,1}, k in {0,1,2,3}
    #   C[0,0]=idx0, C[0,1]=idx1
    #   C[1,0]=idx2, C[1,1]=idx3
    #   C[2,0]=idx4, C[2,1]=idx5
    #   C[3,0]=idx6, C[3,1]=idx7
    
    # 2x4x4 = A_left @ B_top + A_right @ B_bottom
    # where:
    # A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # B_top = B[0:2, :] (2x4), B_bottom = B[2:4, :] (2x4)
    #
    # A_left @ B_top = 2x2 @ 2x4: split B_top into B_top_left (2x2, cols 0-1) and B_top_right (2x2, cols 2-3)
    #   = A_left @ B_top_left + A_left @ B_top_right
    # A_right @ B_bottom = 2x2 @ 2x4: split B_bottom into B_bottom_left (2x2, cols 0-1) and B_bottom_right (2x2, cols 2-3)
    #   = A_right @ B_bottom_left + A_right @ B_bottom_right
    #
    # So 4 2x2x2 Strassen multiplications:
    # Block 0: A_left @ B_top_left -> contributes to C[:, 0:2]
    # Block 1: A_left @ B_top_right -> contributes to C[:, 2:4]
    # Block 2: A_right @ B_bottom_left -> contributes to C[:, 0:2]
    # Block 3: A_right @ B_bottom_right -> contributes to C[:, 2:4]
    
    # For each 2x2x2 Strassen block, we need to map:
    # - A indices (2x2): which 2x2 submatrix of the 2x4 A
    # - B indices (2x2): which 2x2 submatrix of the 4x4 B
    # - C indices (2x2): which 2x2 submatrix of the 2x4 C
    
    # Block 0: A_left (rows 0-1, cols 0-1 of A), B_top_left (rows 0-1, cols 0-1 of B), C[:, 0:2]
    # A submatrix indices in 2x4 A: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # B submatrix indices in 4x4 B: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C submatrix indices in 2x4 C: C[0,0]=0, C[0,1]=2, C[1,0]=4, C[1,1]=6
    
    # Block 1: A_left (rows 0-1, cols 0-1 of A), B_top_right (rows 0-1, cols 2-3 of B), C[:, 2:4]
    # A submatrix indices in 2x4 A: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # B submatrix indices in 4x4 B: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # C submatrix indices in 2x4 C: C[0,2]=1, C[0,3]=3, C[1,2]=5, C[1,3]=7
    
    # Block 2: A_right (rows 0-1, cols 2-3 of A), B_bottom_left (rows 2-3, cols 0-1 of B), C[:, 0:2]
    # A submatrix indices in 2x4 A: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # B submatrix indices in 4x4 B: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # C submatrix indices in 2x4 C: C[0,0]=0, C[0,1]=2, C[1,0]=4, C[1,1]=6
    
    # Block 3: A_right (rows 0-1, cols 2-3 of A), B_bottom_right (rows 2-3, cols 2-3 of B), C[:, 2:4]
    # A submatrix indices in 2x4 A: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # B submatrix indices in 4x4 B: B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15
    # C submatrix indices in 2x4 C: C[0,2]=1, C[0,3]=3, C[1,2]=5, C[1,3]=7
    
    # Define the 4 blocks for 2x4x4
    # Each block: (a_indices, b_indices, c_indices)
    # a_indices: [a00, a01, a10, a11] in the 2x4 A
    # b_indices: [b00, b01, b10, b11] in the 4x4 B
    # c_indices: [c00, c01, c10, c11] in the 2x4 C (note: C index = k*2+i)
    
    blocks_2x4x4 = [
        # Block 0
        ([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 4, 6]),
        # Block 1
        ([0, 1, 4, 5], [2, 3, 6, 7], [1, 3, 5, 7]),
        # Block 2
        ([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 4, 6]),
        # Block 3
        ([2, 3, 6, 7], [10, 11, 14, 15], [1, 3, 5, 7]),
    ]
    
    # Now build 2x4x8 by combining two 2x4x4 blocks
    # A is 2x4: indices i*4+j, i in {0,1}, j in {0,1,2,3}
    #   Same as above
    # B is 4x8: indices j*8+k, j in {0,1,2,3}, k in {0,1,2,3,4,5,6,7}
    #   B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[0,4]=4, B[0,5]=5, B[0,6]=6, B[0,7]=7
    #   B[1,0]=8, B[1,1]=9, B[1,2]=10, B[1,3]=11, B[1,4]=12, B[1,5]=13, B[1,6]=14, B[1,7]=15
    #   B[2,0]=16, B[2,1]=17, B[2,2]=18, B[2,3]=19, B[2,4]=20, B[2,5]=21, B[2,6]=22, B[2,7]=23
    #   B[3,0]=24, B[3,1]=25, B[3,2]=26, B[3,3]=27, B[3,4]=28, B[3,5]=29, B[3,6]=30, B[3,7]=31
    # C is 2x8: indices k*2+i, i in {0,1}, k in {0,1,2,3,4,5,6,7}
    #   C[0,0]=0, C[0,1]=1
    #   C[1,0]=2, C[1,1]=3
    #   C[2,0]=4, C[2,1]=5
    #   C[3,0]=6, C[3,1]=7
    #   C[4,0]=8, C[4,1]=9
    #   C[5,0]=10, C[5,1]=11
    #   C[6,0]=12, C[6,1]=13
    #   C[7,0]=14, C[7,1]=15
    
    # Left half: B_left = B[:, 0:4] (4x4), C_left = C[:, 0:4] (2x4)
    # Right half: B_right = B[:, 4:8] (4x4), C_right = C[:, 4:8] (2x4)
    
    # For left half (B indices offset by 0):
    # B_left indices in 4x8 B:
    #   B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    #   B[1,0]=8, B[1,1]=9, B[1,2]=10, B[1,3]=11
    #   B[2,0]=16, B[2,1]=17, B[2,2]=18, B[2,3]=19
    #   B[3,0]=24, B[3,1]=25, B[3,2]=26, B[3,3]=27
    # C_left indices in 2x8 C:
    #   C[0,0]=0, C[0,1]=2, C[1,0]=4, C[1,1]=6, C[0,2]=1, C[0,3]=3, C[1,2]=5, C[1,3]=7
    
    # For right half (B indices offset by 4 in column):
    # B_right indices in 4x8 B:
    #   B[0,4]=4, B[0,5]=5, B[0,6]=6, B[0,7]=7
    #   B[1,4]=12, B[1,5]=13, B[1,6]=14, B[1,7]=15
    #   B[2,4]=20, B[2,5]=21, B[2,6]=22, B[2,7]=23
    #   B[3,4]=28, B[3,5]=29, B[3,6]=30, B[3,7]=31
    # C_right indices in 2x8 C:
    #   C[0,4]=8, C[0,5]=10, C[1,4]=12, C[1,5]=14, C[0,6]=9, C[0,7]=11, C[1,6]=13, C[1,7]=15
    
    # A indices in 2x4 A (same for both halves):
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    #   A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    
    # Build blocks for 2x4x8
    # Each block: (a_indices, b_indices, c_indices)
    # 8 Strassen blocks total (4 for left, 4 for right)
    
    blocks_2x4x8 = []
    
    # Left half blocks (B columns 0-3)
    # Block 0L: A_left @ B_top_left
    blocks_2x4x8.append(([0, 1, 4, 5], [0, 1, 8, 9], [0, 2, 4, 6]))
    # Block 1L: A_left @ B_top_right (B cols 2-3)
    blocks_2x4x8.append(([0, 1, 4, 5], [2, 3, 10, 11], [1, 3, 5, 7]))
    # Block 2L: A_right @ B_bottom_left
    blocks_2x4x8.append(([2, 3, 6, 7], [16, 17, 24, 25], [0, 2, 4, 6]))
    # Block 3L: A_right @ B_bottom_right (B cols 2-3)
    blocks_2x4x8.append(([2, 3, 6, 7], [18, 19, 26, 27], [1, 3, 5, 7]))
    
    # Right half blocks (B columns 4-7)
    # Block 0R: A_left @ B_top_left (B cols 4-5)
    blocks_2x4x8.append(([0, 1, 4, 5], [4, 5, 12, 13], [8, 10, 12, 14]))
    # Block 1R: A_left @ B_top_right (B cols 6-7)
    blocks_2x4x8.append(([0, 1, 4, 5], [6, 7, 14, 15], [9, 11, 13, 15]))
    # Block 2R: A_right @ B_bottom_left (B cols 4-5)
    blocks_2x4x8.append(([2, 3, 6, 7], [20, 21, 28, 29], [8, 10, 12, 14]))
    # Block 3R: A_right @ B_bottom_right (B cols 6-7)
    blocks_2x4x8.append(([2, 3, 6, 7], [22, 23, 30, 31], [9, 11, 13, 15]))
    
    # Total rank = 8 blocks * 7 = 56
    rank = 56
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]  # 2x4 = 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 4x8 = 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 8x2 = 16 rows
    
    # Fill in the Strassen decomposition
    for block_idx, (a_indices, b_indices, c_indices) in enumerate(blocks_2x4x8):
        base_col = block_idx * 7
        for s in range(7):
            col = base_col + s
            # U
            a00, a01, a10, a11 = a_indices
            U[a00][col] += strassen_U[s][0]
            U[a01][col] += strassen_U[s][1]
            U[a10][col] += strassen_U[s][2]
            U[a11][col] += strassen_U[s][3]
            # V
            b00, b01, b10, b11 = b_indices
            V[b00][col] += strassen_V[s][0]
            V[b01][col] += strassen_V[s][1]
            V[b10][col] += strassen_V[s][2]
            V[b11][col] += strassen_V[s][3]
            # W
            c00, c01, c10, c11 = c_indices
            W[c00][col] += strassen_W[s][0]
            W[c01][col] += strassen_W[s][1]
            W[c10][col] += strassen_W[s][2]
            W[c11][col] += strassen_W[s][3]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
