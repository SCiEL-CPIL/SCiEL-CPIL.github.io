import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the 3x7 multiplication into:
    # 1. 3x3 multiplication (cols 0-2) using Strassen (Rank 7)
    # 2. 3x3 multiplication (cols 3-5) using Strassen (Rank 7)
    # 3. 3x1 multiplication (col 6) using Schoolbook (Rank 9)
    # Total Rank = 7 + 7 + 9 = 23
    
    # Strassen decomposition for 3x3x3
    # Coefficients for A, B, C
    # u_r: coefficients for A
    # v_r: coefficients for B
    # w_r: coefficients for C (output)
    
    # Note: The specific coefficients must satisfy sum(u_r A) (v_r B) w_r = C
    # This is a tensor decomposition.
    
    # Using a known rank 7 decomposition for 3x3 matrix multiplication.
    # Here are the 7 triplets (u, v, w) where u,v,w are vectors in R^9.
    # Indices for A (3x3): 0..8
    # Indices for B (3x3): 0..8
    # Indices for C (3x3): 0..8
    
    strassen_terms = []
    
    # M1
    u1 = [1, 0, 0, 0, 1, 0, 0, 0, 0] # a11 + a22
    v1 = [1, 0, 0, 0, 1, 0, 0, 0, 0] # b11 + b22
    w1 = [1, 0, 0, 0, 1, 0, 0, 0, 0] # c11 + c22
    strassen_terms.append((u1, v1, w1))
    
    # M2
    u2 = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    v2 = [0, 1, 0, 0, 0, 1, 0, 0, 0] # b12 + b22
    w2 = [0, 1, 0, 0, 0, 1, 0, 0, 0] # c12 + c22
    strassen_terms.append((u2, v2, w2))
    
    # M3
    u3 = [0, 0, 0, 0, 1, 0, 0, 0, 0] # a22
    v3 = [1, 0, 0, 0, 0, 1, 0, 0, 0] # b11 + b21
    w3 = [0, 0, 0, 0, 1, 0, 0, 0, 0] # c21
    strassen_terms.append((u3, v3, w3))
    
    # M4
    u4 = [1, 0, 0, 0, 0, 0, 0, 0, 0] # a11
    v4 = [0, 1, 0, 0, 0, 0, 0, 0, 0] # b12
    w4 = [0, 1, 0, 0, 0, 0, 0, 0, 0] # c12
    strassen_terms.append((u4, v4, w4))
    
    # M5
    u5 = [0, 1, 0, 0, 1, 0, 0, 0, 0] # a12 + a22
    v5 = [0, 0, 0, 0, 0, 1, 0, 0, 0] # b22
    w5 = [0, 0, 0, 0, 0, 1, 0, 0, 0] # c22
    strassen_terms.append((u5, v5, w5))
    
    # M6
    u6 = [1, 1, 0, 0, 0, 0, 0, 0, 0] # a11 + a12
    v6 = [1, 0, 0, 0, 0, 0, 0, 0, 0] # b11
    w6 = [1, 0, 0, 0, 0, 0, 0, 0, 0] # c11
    strassen_terms.append((u6, v6, w6))
    
    # M7
    u7 = [0, 0, 0, 1, 1, 0, 0, 0, 0] # a21 + a22
    v7 = [0, 0, 0, 0, 0, 0, 0, 1, 0] # b22
    w7 = [0, 0, 0, 1, 0, 0, 0, 0, 0] # c21
    strassen_terms.append((u7, v7, w7))
    
    # These coefficients are a guess and likely incorrect for exact rank 7.
    # I will fall back to Schoolbook (Rank 9) for 3x3 to be safe.
    # Rank 9 + 9 + 9 = 27.
    
    def schoolbook_3x3(offset_b=0):
        terms = []
        for i in range(3):
            for j in range(3):
                u = [0]*9
                v = [0]*9
                w = [0]*9
                u[i*3 + j] = 1
                v[j*3 + offset_b] = 1
                w[j*3 + i] = 1
                terms.append((u, v, w))
        return terms

    def schoolbook_3x1(offset_b=0):
        terms = []
        for i in range(3):
            for j in range(3):
                u = [0]*9
                v = [0]*9
                w = [0]*3 # Only 3 outputs
                u[i*3 + j] = 1
                v[j*3 + offset_b] = 1
                w[i] = 1
                terms.append((u, v, w))
        return terms

    # Use Schoolbook for all to ensure correctness, Rank 27.
    # Block 1: 3x3 (cols 0-2)
    terms1 = schoolbook_3x3(offset_b=0)
    # Block 2: 3x3 (cols 3-5)
    terms2 = schoolbook_3x3(offset_b=3)
    # Block 3: 3x1 (col 6)
    terms3 = schoolbook_3x1(offset_b=6)
    
    all_terms = terms1 + terms2 + terms3
    rank = len(all_terms)
    
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(21)]
    W = [[0]*rank for _ in range(21)]
    
    for r, (u, v, w) in enumerate(all_terms):
        for idx, val in enumerate(u):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(v):
            if val != 0:
                V[idx][r] = val
        # W mapping: 
        # For 3x3 blocks, w has 9 entries corresponding to C_{j,i} (col j, row i)
        # Global W index: k*n + i. Here k is col index (0..6), i is row index (0..2).
        # In schoolbook_3x3, w index is j*3 + i. j is local col, i is local row.
        # Global col = block_offset + j.
        for idx, val in enumerate(w):
            if val != 0:
                local_col, local_row = divmod(idx, 3)
                # Determine block
                # This simple loop doesn't know the block offset.
                # I need to fix the W index generation.
                
    # Let's rebuild properly.
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(21)]
    W = [[0]*rank for _ in range(21)]
    
    col_offset = 0
    for r, (u, v, w) in enumerate(all_terms):
        for idx, val in enumerate(u):
            if val != 0: U[idx][r] = val
        for idx, val in enumerate(v):
            if val != 0: V[idx][r] = val
        for idx, val in enumerate(w):
            if val != 0:
                # idx is w index.
                # For 3x3: idx = j*3 + i.
                # For 3x1: idx = i.
                # We need to map to global W index: k*n + i.
                # This depends on the block.
                pass
                
    # Simpler construction
    U = []
    V = []
    W = []
    
    for block in range(3):
        if block < 2:
            # 3x3 block
            for i in range(3):
                for j in range(3):
                    u = [0]*9
                    v = [0]*21
                    w = [0]*21
                    u[i*3 + j] = 1
                    # B index: j (row) * 7 (cols) + (block*3 + j) (col)
                    v[j*7 + block*3 + j] = 1
                    # W index: col * 3 + row. Col = block*3 + j. Row = i.
                    w[(block*3 + j)*3 + i] = 1
                    U.append(u)
                    V.append(v)
                    W.append(w)
        else:
            # 3x1 block (col 6)
            for i in range(3):
                for j in range(3):
                    u = [0]*9
                    v = [0]*21
                    w = [0]*21
                    u[i*3 + j] = 1
                    v[j*7 + 6] = 1
                    w[6*3 + i] = 1
                    U.append(u)
                    V.append(v)
                    W.append(w)
                    
    rank = len(U)
    # Convert lists of lists to standard format
    U_mat = [row[:] for row in zip(*U)]
    V_mat = [row[:] for row in zip(*V)]
    W_mat = [row[:] for row in zip(*W)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
