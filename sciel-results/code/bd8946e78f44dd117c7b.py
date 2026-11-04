def construct():
    n, m, p = 4, 4, 6
    # We decompose the 4x4x6 multiplication into 2 layers (j=0,1) of 6 independent 2x2x2 multiplications.
    # Each 2x2x2 multiplication uses the schoolbook algorithm (rank 4).
    # Total rank = 2 * 6 * 4 = 48.
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)] # 16 x 48
    V = [[0] * rank for _ in range(m * p)] # 24 x 48
    W = [[0] * rank for _ in range(p * n)] # 24 x 48
    
    col = 0
    # Iterate over the block index j (0 or 1)
    for j in range(2):
        # Iterate over the output block indices i (0 or 1) and k (0, 1, 2)
        for i in range(2):
            for k in range(3):
                # This block corresponds to C[2i:2i+2, 2k:2k+2]
                # It is computed as A[2i:2i+2, 2j:2j+2] * B[2j:2j+2, 2k:2k+2]
                
                # The 4 terms of the 2x2x2 schoolbook multiplication:
                # C_00 = A_00 B_00 + A_01 B_10
                # C_01 = A_00 B_01 + A_01 B_11
                # C_10 = A_10 B_00 + A_11 B_10
                # C_11 = A_10 B_01 + A_11 B_11
                
                # Local indices for A block (i, j):
                # A_local_00 is A[2i, 2j]
                # A_local_01 is A[2i, 2j+1]
                # A_local_10 is A[2i+1, 2j]
                # A_local_11 is A[2i+1, 2j+1]
                
                # Local indices for B block (j, k):
                # B_local_00 is B[2j, 2k]
                # B_local_01 is B[2j, 2k+1]
                # B_local_10 is B[2j+1, 2k]
                # B_local_11 is B[2j+1, 2k+1]
                
                # Local indices for C block (i, k):
                # C_local_00 is C[2i, 2k]
                # C_local_01 is C[2i, 2k+1]
                # C_local_10 is C[2i+1, 2k]
                # C_local_11 is C[2i+1, 2k+1]
                
                # Term 1: A_00 * B_00 -> C_00
                U[(2*i) * 4 + (2*j)][col] = 1
                V[(2*j) * 6 + (2*k)][col] = 1
                W[(2*k) * 4 + (2*i)][col] = 1
                col += 1
                
                # Term 2: A_01 * B_10 -> C_00
                U[(2*i) * 4 + (2*j+1)][col] = 1
                V[(2*j+1) * 6 + (2*k)][col] = 1
                W[(2*k) * 4 + (2*i)][col] = 1
                col += 1
                
                # Term 3: A_00 * B_01 -> C_01
                U[(2*i) * 4 + (2*j)][col] = 1
                V[(2*j) * 6 + (2*k+1)][col] = 1
                W[(2*k+1) * 4 + (2*i)][col] = 1
                col += 1
                
                # Term 4: A_01 * B_11 -> C_01
                U[(2*i) * 4 + (2*j+1)][col] = 1
                V[(2*j+1) * 6 + (2*k+1)][col] = 1
                W[(2*k+1) * 4 + (2*i)][col] = 1
                col += 1
                
                # Term 5: A_10 * B_00 -> C_10
                U[(2*i+1) * 4 + (2*j)][col] = 1
                V[(2*j) * 6 + (2*k)][col] = 1
                W[(2*k) * 4 + (2*i+1)][col] = 1
                col += 1
                
                # Term 6: A_11 * B_10 -> C_10
                U[(2*i+1) * 4 + (2*j+1)][col] = 1
                V[(2*j+1) * 6 + (2*k)][col] = 1
                W[(2*k) * 4 + (2*i+1)][col] = 1
                col += 1
                
                # Term 7: A_10 * B_01 -> C_11
                U[(2*i+1) * 4 + (2*j)][col] = 1
                V[(2*j) * 6 + (2*k+1)][col] = 1
                W[(2*k+1) * 4 + (2*i+1)][col] = 1
                col += 1
                
                # Term 8: A_11 * B_11 -> C_11
                U[(2*i+1) * 4 + (2*j+1)][col] = 1
                V[(2*j+1) * 6 + (2*k+1)][col] = 1
                W[(2*k+1) * 4 + (2*i+1)][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
