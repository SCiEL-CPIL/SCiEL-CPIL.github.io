import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # A is 2x2, B is 2x2, C is 2x2
    # U rows: (i,j) of A
    # V rows: (j,k) of B
    # W rows: (k,i) of C
    
    strassen_U = [
        [1, 0, 1, 0, 1, 0, -1], # a00
        [0, 0, 0, 1, 0, -1, 1], # a10
        [1, 0, 0, 0, 1, 0, 0],  # a01? No, standard indexing A[i][j]
        # Let's use standard Strassen coefficients for A[i][j]
    ]
    
    # Let's define Strassen explicitly for A (2x2), B (2x2) -> C (2x2)
    # m1: (a00+a11)(b00+b11)
    # m2: (a10+a11)b00
    # m3: a00(b10-b11)
    # m4: a11(b01-b00)
    # m5: (a00+a01)b11
    # m6: (a10-a00)(b00+b01)
    # m7: (a01-a11)(b10+b11)
    
    # U (A part):
    # Row 0 (a00): 1, 0, 1, 0, 1, -1, 0
    # Row 1 (a10): 0, 1, 0, 0, 0, 1, 0
    # Row 2 (a01): 0, 0, 0, 0, 1, 0, 1
    # Row 3 (a11): 1, 1, 0, 1, 0, 0, -1
    
    # V (B part):
    # Col 0 (b00): 1, 1, 0, -1, 0, 1, 0
    # Col 1 (b10): 0, 0, 1, 0, 0, 0, 1
    # Col 2 (b01): 0, 0, 0, 1, 0, 1, 0
    # Col 3 (b11): 1, 0, -1, 0, 1, 0, 1
    
    # W (C part):
    # C00 = m1 - m4 + m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m6 + m7
    
    # W Row 0 (C00, index k=0,i=0): -1, -1, 0, 1, 1, 0, -1
    # W Row 1 (C10, index k=0,i=1): 0, 1, 0, 1, 0, 0, 0
    # W Row 2 (C01, index k=1,i=0): 0, 0, 1, 0, 1, 0, 0
    # W Row 3 (C11, index k=1,i=1): 1, -1, 0, 0, 0, 1, 1
    
    sU = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    sV = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    sW = [
        [-1, -1, 0, 1, 1, 0, -1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 0, 0, 0, 1, 1]
    ]
    
    # Schoolbook 1x2x2 decomposition (Rank 4)
    # A (1x2), B (2x2) -> C (1x2)
    # U (1x2):
    # Row 0 (a0): 1, 0, 0, 0
    # Row 1 (a1): 0, 0, 1, 1  -- Wait, standard schoolbook:
    # c0 = a0*b00 + a1*b10
    # c1 = a0*b01 + a1*b11
    # m1 = a0*b00
    # m2 = a1*b10
    # m3 = a0*b01
    # m4 = a1*b11
    
    # U:
    # a0: 1, 0, 1, 0
    # a1: 0, 1, 0, 1
    
    # V:
    # b00: 1, 0, 0, 0
    # b10: 0, 1, 0, 0
    # b01: 0, 0, 1, 0
    # b11: 0, 0, 0, 1
    
    # W:
    # c0: 1, 1, 0, 0
    # c1: 0, 0, 1, 1
    
    hU = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    hV = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    hW = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    def add_block(sU, sV, sW, i_off, j_off, k_off, i_size, j_size, k_size, col_start):
        # Map local indices to global
        # sU is i_size x j_size x rank_block
        # sV is j_size x k_size x rank_block
        # sW is k_size x i_size x rank_block
        
        for li in range(i_size):
            for lj in range(j_size):
                gi = i_off + li
                gj = j_off + lj
                global_u_idx = gi * m + gj
                for r in range(len(sU[li][lj])):
                    U[global_u_idx][col_start + r] += sU[li][lj][r]
                    
        for lj in range(j_size):
            for lk in range(k_size):
                gj = j_off + lk
                gk = k_off + lk
                global_v_idx = gj * p + gk
                for r in range(len(sV[lj][lk])):
                    V[global_v_idx][col_start + r] += sV[lj][lk][r]
                    
        for lk in range(k_size):
            for li in range(i_size):
                gk = k_off + lk
                gi = i_off + li
                global_w_idx = gk * n + gi
                for r in range(len(sW[lk][li])):
                    W[global_w_idx][col_start + r] += sW[lk][li][r]

    # Reshape sU, sV, sW to 3D for easier iteration
    # sU: 2x2x7
    sU3 = [[[sU[i][k] for k in range(7)] for i in range(2)] for j in range(2)] # Wrong, sU is 2x2 matrix of vectors?
    # sU is list of rows. Each row is a vector of length 7.
    # sU[i][j] is not defined like that.
    # sU is 4x7. Rows 0..3 correspond to (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)?
    # In my definition:
    # Row 0: a00 (i=0, j=0)
    # Row 1: a10 (i=1, j=0)
    # Row 2: a01 (i=0, j=1)
    # Row 3: a11 (i=1, j=1)
    
    # So sU_flat[r] corresponds to A[ri][rj].
    # ri = r % 2? No.
    # r=0 -> (0,0)
    # r=1 -> (1,0)
    # r=2 -> (0,1)
    # r=3 -> (1,1)
    
    # This mapping is non-standard. Let's create a helper to extract.
    
    def get_block_factors(local_U, local_V, local_W, i_off, j_off, k_off, i_sz, j_sz, k_sz, col_start):
        # local_U is i_sz * j_sz list of vectors
        # local_V is j_sz * k_sz list of vectors
        # local_W is k_sz * i_sz list of vectors
        
        for li in range(i_sz):
            for lj in range(j_sz):
                vec_U = local_U[li][lj]
                gi = i_off + li
                gj = j_off + lj
                idx_U = gi * m + gj
                for r, val in enumerate(vec_U):
                    U[idx_U][col_start + r] += val
                    
        for lj in range(j_sz):
            for lk in range(k_sz):
                vec_V = local_V[lj][lk]
                gj = j_off + lj
                gk = k_off + lk
                idx_V = gj * p + gk
                for r, val in enumerate(vec_V):
                    V[idx_V][col_start + r] += val
                    
        for lk in range(k_sz):
            for li in range(i_sz):
                vec_W = local_W[lk][li]
                gk = k_off + lk
                gi = i_off + li
                idx_W = gk * n + gi
                for r, val in enumerate(vec_W):
                    W[idx_W][col_start + r] += val

    # Prepare 3D arrays for Strassen
    sU_3d = [[0]*7 for _ in range(4)] # 2x2
    sU_3d[0][0] = sU[0] # a00
    sU_3d[1][0] = sU[1] # a10
    sU_3d[0][1] = sU[2] # a01
    sU_3d[1][1] = sU[3] # a11
    
    sV_3d = [[0]*7 for _ in range(4)] # 2x2
    sV_3d[0][0] = sV[0] # b00
    sV_3d[0][1] = sV[1] # b10
    sV_3d[1][0] = sV[2] # b01
    sV_3d[1][1] = sV[3] # b11
    
    sW_3d = [[0]*7 for _ in range(4)] # 2x2 (k,i)
    sW_3d[0][0] = sW[0] # c00
    sW_3d[0][1] = sW[1] # c10
    sW_3d[1][0] = sW[2] # c01
    sW_3d[1][1] = sW[3] # c11
    
    # Prepare 3D arrays for Schoolbook 1x2x2
    hU_3d = [[0]*4 for _ in range(2)] # 1x2
    hU_3d[0][0] = hU[0] # a0
    hU_3d[0][1] = hU[1] # a1
    
    hV_3d = [[0]*4 for _ in range(4)] # 2x2
    hV_3d[0][0] = hV[0] # b00
    hV_3d[0][1] = hV[1] # b10
    hV_3d[1][0] = hV[2] # b01
    hV_3d[1][1] = hV[3] # b11
    
    hW_3d = [[0]*4 for _ in range(2)] # 2x1 (k,i)
    hW_3d[0][0] = hW[0] # c0
    hW_3d[1][0] = hW[1] # c1
    
    # 1. Top-Left: A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    # Term 1: i_off=0, j_off=0, k_off=0. Strassen.
    get_block_factors(sU_3d, sV_3d, sW_3d, 0, 0, 0, 2, 2, 2, col)
    col += 7
    # Term 2: i_off=0, j_off=2, k_off=0. Strassen.
    get_block_factors(sU_3d, sV_3d, sW_3d, 0, 2, 0, 2, 2, 2, col)
    col += 7
    
    # 2. Top-Right: A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4]
    # Term 3: i_off=0, j_off=0, k_off=2. Strassen.
    get_block_factors(sU_3d, sV_3d, sW_3d, 0, 0, 2, 2, 2, 2, col)
    col += 7
    # Term 4: i_off=0, j_off=2, k_off=2. Strassen.
    get_block_factors(sU_3d, sV_3d, sW_3d, 0, 2, 2, 2, 2, 2, col)
    col += 7
    
    # 3. Bottom-Left: A[2:3, 0:2] * B[0:2, 0:2] + A[2:3, 2:4] * B[2:4, 0:2]
    # Term 5: i_off=2, j_off=0, k_off=0. Schoolbook 1x2x2.
    get_block_factors(hU_3d, hV_3d, hW_3d, 2, 0, 0, 1, 2, 2, col)
    col += 4
    # Term 6: i_off=2, j_off=2, k_off=0. Schoolbook 1x2x2.
    get_block_factors(hU_3d, hV_3d, hW_3d, 2, 2, 0, 1, 2, 2, col)
    col += 4
    
    # 4. Bottom-Right: A[2:3, 0:2] * B[0:2, 2:4] + A[2:3, 2:4] * B[2:4, 2:4]
    # Term 7: i_off=2, j_off=0, k_off=2. Schoolbook 1x2x2.
    get_block_factors(hU_3d, hV_3d, hW_3d, 2, 0, 2, 1, 2, 2, col)
    col += 4
    # Term 8: i_off=2, j_off=2, k_off=2. Schoolbook 1x2x2.
    get_block_factors(hU_3d, hV_3d, hW_3d, 2, 2, 2, 1, 2, 2, col)
    col += 4
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
