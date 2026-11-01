import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    
    # Rank 42 solution
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen terms for 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A B
    
    # Term 0: (a00+a11)(b00+b11)
    # Term 1: (a10+a11)b00
    # Term 2: a00(b10-b11)
    # Term 3: a11(b01-b00)
    # Term 4: (a00+a01)b11
    # Term 5: (a01-a00)(b00+b10)
    # Term 6: (a10-a11)(b01+b11)
    
    # Coefficients for A part (rows 0,1 of 2x2 block)
    # Index 0: a00, 1: a01, 2: a10, 3: a11
    strassen_A = [
        [1, 0, 0, 1], # t0
        [0, 0, 1, 1], # t1
        [1, 0, 0, 0], # t2
        [0, 0, 0, 1], # t3
        [1, 1, 0, 0], # t4
        [-1, 1, 0, 0],# t5
        [0, 0, 1, -1] # t6
    ]
    
    # Coefficients for B part (rows 0,1 of 2x2 block)
    # Index 0: b00, 1: b01, 2: b10, 3: b11
    strassen_B = [
        [1, 0, 0, 1], # t0
        [1, 0, 0, 0], # t1
        [0, -1, 1, 0],# t2
        [-1, 1, 0, 0],# t3
        [0, 0, 0, 1], # t4
        [1, 0, 1, 0], # t5
        [0, 1, 0, 1]  # t6
    ]
    
    # Coefficients for C part
    # C00 = t0 + t3 - t4 + t6
    # C01 = t2 + t4
    # C10 = t1 + t3
    # C11 = t0 - t1 + t2 + t5
    # Index 0: c00, 1: c01, 2: c10, 3: c11
    strassen_C = [
        [1, 0, 0, 1], # t0
        [0, 0, 1, -1],# t1
        [0, 1, 0, 1], # t2
        [1, 0, 1, 0], # t3
        [-1, 1, 0, 0],# t4
        [0, 0, 0, 1], # t5
        [1, 0, 0, 0]  # t6
    ]
    
    # Mapping local 2x2 indices to global indices
    # A global: i in 0..1, j in 0..3
    # B global: j in 0..3, k in 0..5
    # C global: i in 0..1, k in 0..5
    
    for g in range(3): # 3 groups of 2 columns in B
        # k range: 2g, 2g+1
        k_start = 2 * g
        
        for s in range(2): # 2 sub-blocks in A (Left, Right)
            # A columns: 2s, 2s+1
            a_col_start = 2 * s
            
            # B rows: 2s, 2s+1
            b_row_start = 2 * s
            
            for t in range(7):
                col = g * 14 + s * 7 + t
                
                # Process A
                for local_i in range(2):
                    for local_j in range(2):
                        coeff = strassen_A[t][local_i * 2 + local_j]
                        if coeff != 0:
                            global_i = local_i
                            global_j = a_col_start + local_j
                            idx = global_i * m + global_j
                            U[idx][col] += coeff
                
                # Process B
                for local_i in range(2):
                    for local_j in range(2):
                        coeff = strassen_B[t][local_i * 2 + local_j]
                        if coeff != 0:
                            global_j = b_row_start + local_i
                            global_k = k_start + local_j
                            idx = global_j * p + global_k
                            V[idx][col] += coeff
                
                # Process C
                for local_i in range(2):
                    for local_j in range(2):
                        coeff = strassen_C[t][local_i * 2 + local_j]
                        if coeff != 0:
                            global_i = local_i
                            global_k = k_start + local_j
                            idx = global_k * n + global_i
                            W[idx][col] += coeff

    return {"U": U, "V": V, "W": W}
