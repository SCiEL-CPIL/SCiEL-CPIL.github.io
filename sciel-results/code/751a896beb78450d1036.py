import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Total rank
    rank_444 = 56  # 8 * 7 (Strassen)
    rank_442 = 32  # 4 * 2 * 4 (Schoolbook)
    rank_416 = 24  # 4 * 6 (Schoolbook)
    total_rank = rank_444 + rank_442 + rank_416
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(n * m):
            if u_vec[i] != 0:
                U[i][col] += u_vec[i]
        for j in range(m * p):
            if v_vec[j] != 0:
                V[j][col] += v_vec[j]
        for k in range(p * n):
            if w_vec[k] != 0:
                W[k][col] += w_vec[k]
        col += 1

    # Strassen coefficients for 2x2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (in W: k*n+i, so 00->0, 01->1, 10->2, 11->3 for n=2)
    
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # C21 += M2
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # C12 += M3
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 1]), # C11 += M4, C21 += M4
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]), # C11 -= M5, C22 += M5
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # C22 += M6
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]), # C11 += M7, C22 -= M7
    ]
    
    # Wait, I need to double check the Strassen W coefficients.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's re-verify M4.
    # M4 = A22 * (B21 - B11)
    # C11 gets +M4. C21 gets +M4.
    # So w for M4 should be [1, 0, 1, 0] in C-index (00, 01, 10, 11).
    # In W-index (k*n+i):
    # C00 (0,0) -> idx 0
    # C01 (0,1) -> idx 1
    # C10 (1,0) -> idx 2
    # C11 (1,1) -> idx 3
    # So w = [1, 0, 1, 0].
    
    # Let's re-verify M5.
    # M5 = (A11 + A12) * B22
    # C11 gets -M5. C12 gets +M5.
    # w = [0, 1, 0, -1].
    
    # Let's re-verify M7.
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 gets +M7. C22 gets -M7.
    # w = [1, 0, 0, -1].
    
    # Corrected Strassen Terms (u, v, w)
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A10+A11)B00 -> C10 += 1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3: A00(B01-B11) -> C01 += 1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4: A11(B10-B00) -> C00 += 1, C10 += 1
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5: (A00+A01)B11 -> C01 += 1, C00 -= 1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: (A10-A00)(B00+B01) -> C11 += 1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
    ]

    # 1. 4x4x4 Part (A1 * B2a)
    # A1 is 4x4, B2a is 4x4.
    # Blocks: A11, A12, A21, A22 (2x2 each)
    # B21, B22, B23, B24 (2x2 each) -- Wait, B is 4x6.
    # B2a is the first 4 columns of B. So B2a is 4x4.
    # Blocks of B2a: B11, B12, B21, B22 (2x2 each).
    
    # We need to compute:
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # Indices in 4x4:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    # For each product X Y, we add 7 terms.
    # X is a 2x2 block of A. Y is a 2x2 block of B.
    # The result is added to a 2x2 block of C.
    
    products = [
        # (A_block, B_block, C_block)
        # C11 = A11 B11 + A12 B21
        ((0,0), (0,0), (0,0)), # A11 B11 -> C11
        ((0,2), (2,0), (0,0)), # A12 B21 -> C11
        # C12 = A11 B12 + A12 B22
        ((0,0), (0,2), (0,2)), # A11 B12 -> C12
        ((0,2), (2,2), (0,2)), # A12 B22 -> C12
        # C21 = A21 B11 + A22 B21
        ((2,0), (0,0), (2,0)), # A21 B11 -> C21
        ((2,2), (2,0), (2,0)), # A22 B21 -> C21
        # C22 = A21 B12 + A22 B22
        ((2,0), (0,2), (2,2)), # A21 B12 -> C22
        ((2,2), (2,2), (2,2)), # A22 B22 -> C22
    ]
    
    for a_r, a_c, b_r, b_c, c_r, c_c in products:
        for u2, v2, w2 in strassen_terms:
            # Map 2x2 indices to 4x4 indices
            u4 = [0] * 16
            v4 = [0] * 16
            w4 = [0] * 16
            
            for i in range(2):
                for j in range(2):
                    # A index: (a_r + i) * 4 + (a_c + j)
                    # But U is indexed by i*m + j. m=5.
                    # Wait, A is 4x5.
                    # A1 is the first 4 columns.
                    # So A1 indices are 0..3 for rows, 0..3 for cols.
                    # U index: row * 5 + col.
                    
                    # B is 5x6.
                    # B2a is rows 0..3, cols 0..3.
                    # V index: row * 6 + col.
                    
                    # C is 4x6.
                    # C11 is rows 0..1, cols 0..1.
                    # W index: col * 4 + row.
                    
                    # u2[i*2+j]
                    u_idx = (a_r + i) * 5 + (a_c + j)
                    u4[u_idx] = u2[i * 2 + j]
                    
                    v_idx = (b_r + i) * 6 + (b_c + j)
                    v4[v_idx] = v2[i * 2 + j]
                    
                    w_idx = (c_c + j) * 4 + (c_r + i)
                    w4[w_idx] = w2[i * 2 + j]
            
            # Extend to full size
            u_full = [0] * 20
            v_full = [0] * 30
            w_full = [0] * 24
            
            for idx in range(16):
                u_full[idx] = u4[idx]
                v_full[idx] = v4[idx]
                w_full[idx] = w4[idx]
            
            add_term(u_full, v_full, w_full)

    # 2. 4x4x2 Part (A1 * B2b)
    # A1 is 4x4. B2b is 4x2 (cols 4,5 of B).
    # C is 4x2 (cols 4,5 of C).
    # C_i,k = sum_j A1_i,j B2b_j,k
    # Schoolbook: 4 * 4 * 2 = 32 terms.
    
    for i in range(4):
        for k in range(2):
            for j in range(4):
                u_full = [0] * 20
                v_full = [0] * 30
                w_full = [0] * 24
                
                u_idx = i * 5 + j
                u_full[u_idx] = 1
                
                v_idx = j * 6 + (4 + k) # B2b cols are 4, 5
                v_full[v_idx] = 1
                
                w_idx = (4 + k) * 4 + i # C cols are 4, 5
                w_full[w_idx] = 1
                
                add_term(u_full, v_full, w_full)

    # 3. 4x1x6 Part (a5 * b5)
    # a5 is 4x1 (col 4 of A).
    # b5 is 1x6 (row 4 of B).
    # C is 4x6.
    # C_i,k = a5_i * b5_k
    # Schoolbook: 4 * 1 * 6 = 24 terms.
    
    for i in range(4):
        for k in range(6):
            u_full = [0] * 20
            v_full = [0] * 30
            w_full = [0] * 24
            
            u_idx = i * 5 + 4 # A col 4
            u_full[u_idx] = 1
            
            v_idx = 4 * 6 + k # B row 4
            v_full[v_idx] = 1
            
            w_idx = k * 4 + i
            w_full[w_idx] = 1
            
            add_term(u_full, v_full, w_full)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
