def construct():
    import time
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill

    n, m, p = 4, 4, 5
    
    # Rank 65 decomposition
    # Part 1: 4x4x4 multiplication using rank-49 (Kronecker product of 2x2x2 Strassen)
    # Part 2: 4x4x1 multiplication using rank-16 (Schoolbook)
    
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 (First 4 columns of B) ---
    # Strassen 2x2x2 terms (7 terms)
    # U, V, W vectors for 2x2x2
    # Indices: U: i*2+j, V: j*2+k, W: k*2+i
    
    S_U = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    S_V = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1], # P3
        [0, 0, 1, -1], # P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    S_W = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 0], # P2
        [0, 1, 0, -1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0], # P5
        [1, 0, 0, -1], # P6
        [-1, 0, 0, 0]  # P7
    ]
    
    # Generate 49 terms for 4x4x4
    for a in range(7):
        for b in range(7):
            idx = a * 7 + b
            
            # U part: size 16
            # Global index I*4+J, Local I, J from term a; i, j from term b
            # Wait, U_a corresponds to the first 2x2 block multiplication?
            # Yes, tensor product: T = T_a \otimes T_b
            # U_{ab}[I_global, J_global] = U_a[I, J] * U_b[i, j]
            # I_global = 2*I + i, J_global = 2*J + j
            # Index in U (4x4) = I_global * 4 + J_global
            
            for I in range(2):
                for J in range(2):
                    ua = S_U[a][I * 2 + J]
                    if ua == 0: continue
                    for i in range(2):
                        for j in range(2):
                            ub = S_U[b][i * 2 + j]
                            if ub == 0: continue
                            val = ua * ub
                            row_u = (2 * I + i) * 4 + (2 * J + j)
                            U[row_u][idx] = val
                            
            # V part: size 16 (for 4x4 block of B)
            # V_{ab}[J_global, K_global] = V_a[J, K] * V_b[j, k]
            # J_global = 2*J + j, K_global = 2*K + k
            # Index in V (4x4 part) = J_global * 4 + K_global
            
            for J in range(2):
                for K in range(2):
                    va = S_V[a][J * 2 + K]
                    if va == 0: continue
                    for j in range(2):
                        for k in range(2):
                            vb = S_V[b][j * 2 + k]
                            if vb == 0: continue
                            val = va * vb
                            row_v = (2 * J + j) * 4 + (2 * K + k)
                            V[row_v][idx] = val
                            
            # W part: size 16 (for 4x4 block of C)
            # W_{ab}[K_global, I_global] = W_a[K, I] * W_b[k, i]
            # K_global = 2*K + k, I_global = 2*I + i
            # Index in W (4x4 part) = K_global * 4 + I_global
            
            for K in range(2):
                for I in range(2):
                    wa = S_W[a][K * 2 + I]
                    if wa == 0: continue
                    for k in range(2):
                        for i in range(2):
                            wb = S_W[b][k * 2 + i]
                            if wb == 0: continue
                            val = wa * wb
                            row_w = (2 * K + k) * 4 + (2 * I + i)
                            W[row_w][idx] = val

    # --- Part 2: 4x4x1 (5th column of B) ---
    # Schoolbook for 4 dot products of length 4
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # 16 terms
    for i in range(4):
        for j in range(4):
            idx = 49 + i * 4 + j
            
            row_u = i * 4 + j
            row_v = j * 5 + 4  # B is 4x5, column 4 (0-indexed)
            row_w = 4 * 4 + i  # C is 4x5, column 4 (0-indexed)
            
            U[row_u][idx] = 1
            V[row_v][idx] = 1
            W[row_w][idx] = 1

    return {"U": U, "V": V, "W": W}
