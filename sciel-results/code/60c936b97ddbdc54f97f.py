"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size is 3, so 2x2 blocks
    block_size = 3
    num_blocks = 2
    
    # Rank 27 for 3x3 schoolbook
    rank_block = block_size * block_size * block_size
    # Rank 7 for 2x2 Strassen
    rank_strassen = 7
    rank = rank_strassen * rank_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen decomposition for 2x2
    # u_a[I_i, I_j], v_a[I_j, I_k], w_a[I_k, I_i]
    # Indices 0,1 for blocks
    strassen_u = [
        [[1, 0], [0, 1]], # M1: A00 + A11
        [[0, 0], [1, 1]], # M2: A10 + A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00 + A01
        [[-1, 0], [1, 0]],# M6: A10 - A00
        [[0, 1], [0, -1]] # M7: A01 - A11
    ]
    strassen_v = [
        [[1, 0], [1, 0]], # M1: B00 + B11 (col 0 is B00, col 1 is B10? No, B is 2x2. B[I_j, I_k])
        # B indices: I_j (row), I_k (col).
        # M1: (B00 + B11). B00 is [0,0], B11 is [1,1].
        # So v1[0,0]=1, v1[1,1]=1.
        [[1, 0], [0, 1]],
        [[0, 1], [0, 0]], # M2: B00. v2[0,0]=1.
        # Correction: M2 = (A10+A11)B00. So v2[0,0]=1.
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],# M3: B01 - B11. v3[0,1]=1, v3[1,1]=-1.
        [[-1, 0], [1, 0]],# M4: B10 - B00. v4[1,0]=1, v4[0,0]=-1.
        [[0, 0], [0, 1]], # M5: B11. v5[1,1]=1.
        [[1, 1], [0, 0]], # M6: B00 + B01. v6[0,0]=1, v6[0,1]=1.
        [[0, 1], [0, 1]]  # M7: B10 + B11. v7[1,0]=1, v7[1,1]=1.
    ]
    # Re-verifying Strassen V vectors
    # M1 = (A00+A11)(B00+B11) -> v1: B00=1, B11=1.
    # M2 = (A10+A11)B00 -> v2: B00=1.
    # M3 = A00(B01-B11) -> v3: B01=1, B11=-1.
    # M4 = A11(B10-B00) -> v4: B10=1, B00=-1.
    # M5 = (A00+A01)B11 -> v5: B11=1.
    # M6 = (A10-A00)(B00+B01) -> v6: B00=1, B01=1.
    # M7 = (A01-A11)(B10+B11) -> v7: B10=1, B11=1.
    
    # Corrected strassen_v
    strassen_v = [
        [[1, 0], [0, 1]], # v1
        [[1, 0], [0, 0]], # v2
        [[0, 1], [0, -1]],# v3
        [[-1, 0], [1, 0]],# v4
        [[0, 0], [0, 1]], # v5
        [[1, 1], [0, 0]], # v6
        [[0, 1], [0, 1]]  # v7
    ]
    
    # w_a[I_k, I_i]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_w = [
        [[1, 0], [0, 1]], # w1: C00=1, C11=1
        [[0, -1], [1, 0]],# w2: C10=1, C11=-1 (Note: w[I_k, I_i]. C10 is I_k=1, I_i=0. C11 is I_k=1, I_i=1)
        [[0, 1], [0, 1]], # w3: C01=1, C11=1
        [[1, 0], [1, 0]], # w4: C00=1, C10=1
        [[-1, 0], [1, 0]],# w5: C00=-1, C01=1
        [[0, 0], [0, 1]], # w6: C11=1
        [[1, 0], [0, 0]]  # w7: C00=1
    ]
    
    # Define Schoolbook decomposition for 3x3
    # u'_b[i', j'], v'_b[j', k'], w'_b[k', i']
    # b corresponds to (i', j', k')
    school_u = []
    school_v = []
    school_w = []
    for ip in range(block_size):
        for jp in range(block_size):
            for kp in range(block_size):
                u = [[0]*block_size for _ in range(block_size)]
                v = [[0]*block_size for _ in range(block_size)]
                w = [[0]*block_size for _ in range(block_size)]
                u[ip][jp] = 1
                v[jp][kp] = 1
                w[kp][ip] = 1
                school_u.append(u)
                school_v.append(v)
                school_w.append(w)
                
    # Construct full decomposition
    for a in range(rank_strassen):
        for b in range(rank_block):
            c = a * rank_block + b
            
            # Fill U
            for Ii in range(num_blocks):
                for Ij in range(num_blocks):
                    val_u_a = strassen_u[a][Ii][Ij]
                    if val_u_a == 0:
                        continue
                    for ip in range(block_size):
                        for jp in range(block_size):
                            val_u_b = school_u[b][ip][jp]
                            if val_u_b == 0:
                                continue
                            i = Ii * block_size + ip
                            j = Ij * block_size + jp
                            U[i * m + j][c] = val_u_a * val_u_b
                            
            # Fill V
            for Ij in range(num_blocks):
                for Ik in range(num_blocks):
                    val_v_a = strassen_v[a][Ij][Ik]
                    if val_v_a == 0:
                        continue
                    for jp in range(block_size):
                        for kp in range(block_size):
                            val_v_b = school_v[b][jp][kp]
                            if val_v_b == 0:
                                continue
                            j = Ij * block_size + jp
                            k = Ik * block_size + kp
                            V[j * p + k][c] = val_v_a * val_v_b
                            
            # Fill W
            for Ik in range(num_blocks):
                for Ii in range(num_blocks):
                    val_w_a = strassen_w[a][Ik][Ii]
                    if val_w_a == 0:
                        continue
                    for kp in range(block_size):
                        for ip in range(block_size):
                            val_w_b = school_w[b][kp][ip]
                            if val_w_b == 0:
                                continue
                            k = Ik * block_size + kp
                            i = Ii * block_size + ip
                            W[k * n + i][c] = val_w_a * val_w_b
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
