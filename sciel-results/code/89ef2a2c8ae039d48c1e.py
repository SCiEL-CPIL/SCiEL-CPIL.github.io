import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create a schoolbook decomposition for n x m by m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        u = [[0] * rank for _ in range(n * m)]
        v = [[0] * rank for _ in range(m * p)]
        w = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u[i * m + j][col] = 1
                    v[j * p + k][col] = 1
                    w[k * n + i][col] = 1
                    col += 1
        return u, v, w

    # Helper to create Strassen decomposition for 2x2
    def strassen_2x2():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # m1 = (a+d)(e+h)
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1; w[3][0] = 1
        
        # m2 = (c+d)e
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[2][1] = 1; w[3][1] = -1
        
        # m3 = a(f-h)
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        w[1][2] = 1; w[3][2] = 1
        
        # m4 = d(g-e)
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        w[0][3] = 1; w[2][3] = 1
        
        # m5 = (a+b)h
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[0][4] = -1; w[1][4] = 1
        
        # m6 = (c-d)(f+h)
        u[2][5] = 1; u[3][5] = -1
        v[1][5] = 1; v[3][5] = 1
        w[3][5] = 1
        
        # m7 = (b-d)(e+g)
        u[1][6] = 1; u[3][6] = -1
        v[0][6] = 1; v[2][6] = 1
        w[0][6] = 1
        
        return u, v, w

    # Helper to create Strassen decomposition for 4x4
    def strassen_4x4():
        # Split 4x4 into four 2x2 blocks
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # We need 7 multiplications of 2x2 matrices.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mi is a 2x2 multiplication.
        # We can generate the U, V, W for each Mi and sum them up.
        
        rank = 49
        u = [[0]*rank for _ in range(16)]
        v = [[0]*rank for _ in range(16)]
        w = [[0]*rank for _ in range(16)]
        
        # We need to map the 2x2 Strassen tensors to the 4x4 indices.
        # A indices:
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        
        # B indices:
        # B11: 0,1,4,5
        # B12: 2,3,6,7
        # B21: 8,9,12,13
        # B22: 10,11,14,15
        
        # C indices (W):
        # C11: 0,1,4,5
        # C12: 2,3,6,7
        # C21: 8,9,12,13
        # C22: 10,11,14,15
        
        # Helper to add a 2x2 Strassen decomposition with offsets and signs
        def add_strassen_2x2(u_offsets, v_offsets, w_offsets, rank_offset, signs_u, signs_v, signs_w):
            u2, v2, w2 = strassen_2x2()
            for i in range(4):
                for r in range(7):
                    u[u_offsets[i]][rank_offset + r] += u2[i][r] * signs_u[i]
            for j in range(4):
                for r in range(7):
                    v[v_offsets[j]][rank_offset + r] += v2[j][r] * signs_v[j]
            for k in range(4):
                for r in range(7):
                    w[w_offsets[k]][rank_offset + r] += w2[k][r] * signs_w[k]

        # M1 = (A11+A22)(B11+B22)
        # U rows: A11 (0,1,4,5) + A22 (10,11,14,15)
        # V rows: B11 (0,1,4,5) + B22 (10,11,14,15)
        # W rows: C11 (0,1,4,5) + C22 (10,11,14,15)
        # This requires summing the inputs.
        # We can't just add offsets easily if we need sums.
        # We must manually construct the U, V, W for M1.
        
        # Let's implement a generic "add tensor" function that takes explicit row indices.
        # This is getting complex. 
        # Alternative: Just use Schoolbook for 4x4 (Rank 64).
        # Total rank with Schoolbook 4x4: 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125.
        # This is same as seed.
        
        # Let's try to implement Strassen 4x4 properly.
        # We will build the U, V, W for each of the 7 Strassen steps.
        
        # Step 1: M1 = (A11+A22)(B11+B22)
        # U1: Rows of A11+A22. 
        # A11 indices: 0,1,4,5. A22 indices: 10,11,14,15.
        # We need to form the sum. 
        # In the tensor decomposition, U is a matrix. 
        # For M1, the "A" part is A11+A22.
        # This means for each rank index in M1, the U vector has 1s at A11 and A22 positions.
        
        # Let's define the 7 blocks of U, V, W for 4x4 Strassen.
        # Each block has rank 7.
        
        # Block 1: M1
        # U1: A11 + A22
        # V1: B11 + B22
        # W1: C11 + C22 (with signs later)
        
        # This manual construction is error-prone.
        # I will use the known fact that Strassen 4x4 has rank 49.
        # I will implement a recursive Strassen that returns U, V, W.
        
        def strassen_recursive(n):
            if n == 2:
                return strassen_2x2()
            
            # Split n into n/2
            half = n // 2
            # Get 2x2 Strassen
            u2, v2, w2 = strassen_2x2()
            
            # We need to compose this with the sub-blocks.
            # This requires building a larger U, V, W.
            # Total rank = 7 * (rank of 2x2 Strassen for size half)
            # Wait, Strassen 2x2 uses 7 multiplications of size half.
            # So total rank = 7 * rank(half).
            
            rank = 7 * (half * half * half) # If using schoolbook for half? No.
            # We should recurse.
            # For n=4, half=2.
            # rank = 7 * 7 = 49.
            
            u_half, v_half, w_half = strassen_recursive(half)
            rank_half = len(u_half[0])
            
            rank = 7 * rank_half
            u = [[0]*rank for _ in range(n*n)]
            v = [[0]*rank for _ in range(n*n)]
            w = [[0]*rank for _ in range(n*n)]
            
            # Define the 7 Strassen products
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B12-B22)
            # M4 = A22(B21-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # We need to map the local 2x2 indices to the global n x n indices.
            # A11: rows 0-1, cols 0-1.
            # A12: rows 0-1, cols 2-3.
            # A21: rows 2-3, cols 0-1.
            # A22: rows 2-3, cols 2-3.
            
            # Helper to get global index for local 2x2 block
            def get_idx(block, i, j):
                # block: 0=A11, 1=A12, 2=A21, 3=A22
                r, c = divmod(block, 2)
                base_r = r * half
                base_c = c * half
                return (base_r + i) * n + (base_c + j)
            
            # This is still complex to wire the sums/differences.
            # Given the time, I will fall back to Schoolbook 4x4 (Rank 64) 
            # and optimize the other blocks.
            # Rank 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125.
            # This is not an improvement.
            
            # I must implement Strassen 4x4.
            # Let's try a different strategy: 
            # Use the 3+2 split with Schoolbook 3x3 (27) and Strassen 2x2 (7).
            # Rank 124.
            # This is an improvement.
            pass

    # I will implement the 3+2 split with Schoolbook 3x3 and Strassen 2x2.
    # Rank 124.
    
    def strassen_2x2():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1; w[3][0] = 1
        
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[2][1] = 1; w[3][1] = -1
        
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        w[1][2] = 1; w[3][2] = 1
        
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        w[0][3] = 1; w[2][3] = 1
        
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[0][4] = -1; w[1][4] = 1
        
        u[2][5] = 1; u[3][5] = -1
        v[1][5] = 1; v[3][5] = 1
        w[3][5] = 1
        
        u[1][6] = 1; u[3][6] = -1
        v[0][6] = 1; v[2][6] = 1
        w[0][6] = 1
        
        return u, v, w

    def schoolbook(n, m, p):
        rank = n * m * p
        u = [[0] * rank for _ in range(n * m)]
        v = [[0] * rank for _ in range(m * p)]
        w = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u[i * m + j][col] = 1
                    v[j * p + k][col] = 1
                    w[k * n + i][col] = 1
                    col += 1
        return u, v, w

    # 3+2 Split
    # A11 (3x3), A12 (3x2), A21 (2x3), A22 (2x2)
    # B11 (3x3), B12 (3x2), B21 (2x3), B22 (2x2)
    
    # 1. A11 B11 (3x3x3) -> Schoolbook 27
    u1, v1, w1 = schoolbook(3, 3, 3)
    # 2. A12 B21 (3x2x3) -> Schoolbook 18
    u2, v2, w2 = schoolbook(3, 2, 3)
    # 3. A11 B12 (3x3x2) -> Schoolbook 18
    u3, v3, w3 = schoolbook(3, 3, 2)
    # 4. A12 B22 (3x2x2) -> Schoolbook 12
    u4, v4, w4 = schoolbook(3, 2, 2)
    # 5. A21 B11 (2x3x3) -> Schoolbook 18
    u5, v5, w5 = schoolbook(2, 3, 3)
    # 6. A22 B21 (2x2x3) -> Schoolbook 12
    u6, v6, w6 = schoolbook(2, 2, 3)
    # 7. A21 B12 (2x3x2) -> Schoolbook 12
    u7, v7, w7 = schoolbook(2, 3, 2)
    # 8. A22 B22 (2x2x2) -> Strassen 7
    u8, v8, w8 = strassen_2x2()
    
    total_rank = 27 + 18 + 18 + 12 + 18 + 12 + 12 + 7
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    def merge(u_local, v_local, w_local, u_map, v_map, w_map, rank_offset):
        for i, local_i in enumerate(u_map):
            for r in range(len(u_local[i])):
                U[local_i][rank_offset + r] = u_local[i][r]
        for j, local_j in enumerate(v_map):
            for r in range(len(v_local[j])):
                V[local_j][rank_offset + r] = v_local[j][r]
        for k, local_k in enumerate(w_map):
            for r in range(len(w_local[k])):
                W[local_k][rank_offset + r] = w_local[k][r]
                
    # Maps for 3x3 block (top-left)
    u_map_33 = [i*5+j for i in range(3) for j in range(3)]
    v_map_33 = [i*5+j for i in range(3) for j in range(3)]
    w_map_33 = [i*5+j for i in range(3) for j in range(3)] # C11: k*5+i. k in 0-2, i in 0-2.
    
    # Maps for 3x2 block (top-right of A, top-right of B)
    # A12: rows 0-2, cols 3-4.
    u_map_32 = [i*5+j for i in range(3) for j in range(3,5)]
    # B21: rows 3-4, cols 0-2.
    v_map_23 = [i*5+j for i in range(3,5) for j in range(3)]
    # C11: rows 0-2, cols 0-2.
    w_map_33_c11 = [i*5+j for i in range(3) for j in range(3)] # k*5+i
    
    # Wait, the W index is k*5+i.
    # For C11, i in 0-2, k in 0-2.
    # So w_map_33_c11 should be [k*5+i for k in range(3) for i in range(3)]?
    # No, the local w_map corresponds to the local W matrix rows.
    # Local W for 3x3x3 has 9 rows.
    # Local row index l corresponds to (k, i) where l = k*3 + i?
    # In schoolbook, w[k*n+i] is used. So local row k*3+i.
    # So for C11, global index is k*5+i.
    # So w_map_33_c11 = [k*5+i for k in range(3) for i in range(3)]
    
    # Let's re-verify the schoolbook W index.
    # schoolbook(3,3,3): w[k*3+i].
    # So local row k*3+i.
    # Global C11 index k*5+i.
    # So w_map_33_c11 = [k*5+i for k in range(3) for i in range(3)]
    
    # 1. A11 B11
    merge(u1, v1, w1, u_map_33, v_map_33, w_map_33_c11, 0)
    
    # 2. A12 B21
    # A12: 3x2. Local u rows 0-5.
    # B21: 2x3. Local v rows 0-5.
    # C11: 3x3. Local w rows 0-8.
    # u_map_32 = [i*5+j for i in range(3) for j in range(3,5)]
    # v_map_23 = [i*5+j for i in range(3,5) for j in range(3)]
    # w_map_33_c11 = [k*5+i for k in range(3) for i in range(3)]
    merge(u2, v2, w2, u_map_32, v_map_23, w_map_33_c11, 27)
    
    # 3. A11 B12
    # A11: 3x3.
    # B12: 3x2. Local v rows 0-5.
    # C12: 3x2. Local w rows 0-5.
    # C12 global: rows 0-2, col 4. Index 4*5+i = 20+i.
    # Local w for 3x3x2: w[k*3+i]. k in 0-1, i in 0-2.
    # So local row k*3+i.
    # Global index 20+i.
    # w_map_32_c12 = [20+i for k in range(2) for i in range(3)]
    # v_map_32 = [i*5+j for i in range(3) for j in range(3,5)]
    merge(u3, v3, w3, u_map_33, v_map_32, w_map_32_c12, 27+18)
    
    # 4. A12 B22
    # A12: 3x2.
    # B22: 2x2.
    # C12: 3x2.
    merge(u4, v4, w4, u_map_32, v_map_22, w_map_32_c12, 27+18+18)
    # v_map_22 = [i*5+j for i in range(3,5) for j in range(3,5)]
    
    # 5. A21 B11
    # A21: 2x3.
    # B11: 3x3.
    # C21: 2x3.
    # C21 global: rows 3-4, cols 0-2. Index k*5+4.
    # Local w for 2x3x3: w[k*2+i]. k in 0-2, i in 0-1.
    # Global index k*5+4.
    # w_map_23_c21 = [k*5+4 for k in range(3) for i in range(2)]
    # u_map_23 = [i*5+j for i in range(3,5) for j in range(3)]
    merge(u5, v5, w5, u_map_23, v_map_33, w_map_23_c21, 27+18+18+12)
    
    # 6. A22 B21
    # A22: 2x2.
    # B21: 2x3.
    # C21: 2x3.
    merge(u6, v6, w6, u_map_22, v_map_23, w_map_23_c21, 27+18+18+12+18)
    # u_map_22 = [i*5+j for i in range(3,5) for j in range(3,5)]
    
    # 7. A21 B12
    # A21: 2x3.
    # B12: 3x2.
    # C22: 2x2.
    # C22 global: rows 3-4, cols 3-4. Index k*5+4. k in 3-4? No.
    # C22 is 2x2. Rows 3-4, Cols 3-4.
    # Global index k*5+i. k in 3-4, i in 3-4.
    # Local w for 2x3x2: w[k*2+i]. k in 0-1, i in 0-1.
    # Global index (3+k)*5 + (3+i) = 15+5k + 3+i = 18+5k+i.
    # w_map_22_c22 = [18+5*k+i for k in range(2) for i in range(2)]
    merge(u7, v7, w7, u_map_23, v_map_32, w_map_22_c22, 27+18+18+12+18+12)
    
    # 8. A22 B22
    # A22: 2x2.
    # B22: 2x2.
    # C22: 2x2.
    merge(u8, v8, w8, u_map_22, v_map_22, w_map_22_c22, 27+18+18+12+18+12+12)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
