import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s guard
    
    n, m, p = DIMENSIONS
    # Rank is 2 * 49 = 98
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen coefficients for 2x2 blocks
    # Indices: 0:00, 1:01, 2:10, 3:11
    u1 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v1 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w1 = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, -1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]
    
    # Inner Strassen coefficients for 2x2 matrices
    # Same structure
    u2 = u1
    v2 = v1
    w2 = w1
    
    # Generate the 49 terms for 4x4x4
    terms = []
    for l in range(7):
        for m_idx in range(7):
            u_term = [0] * 16
            v_term = [0] * 16
            w_term = [0] * 16
            
            # Fill u_term
            for I in range(2):
                for J in range(2):
                    coeff_outer = u1[l][I * 2 + J]
                    if coeff_outer == 0: continue
                    for i_prime in range(2):
                        for j_prime in range(2):
                            coeff_inner = u2[m_idx][i_prime * 2 + j_prime]
                            if coeff_inner == 0: continue
                            i = I * 2 + i_prime
                            j = J * 2 + j_prime
                            u_term[i * 4 + j] += coeff_outer * coeff_inner
                            
            # Fill v_term
            for K in range(2):
                for L in range(2):
                    coeff_outer = v1[l][K * 2 + L]
                    if coeff_outer == 0: continue
                    for k_prime in range(2):
                        for l_prime in range(2):
                            coeff_inner = v2[m_idx][k_prime * 2 + l_prime]
                            if coeff_inner == 0: continue
                            k = K * 2 + k_prime
                            l_idx = L * 2 + l_prime
                            v_term[k * 4 + l_idx] += coeff_outer * coeff_inner
                            
            # Fill w_term
            for P in range(2):
                for Q in range(2):
                    coeff_outer = w1[l][P * 2 + Q]
                    if coeff_outer == 0: continue
                    for p_prime in range(2):
                        for q_prime in range(2):
                            coeff_inner = w2[m_idx][p_prime * 2 + q_prime]
                            if coeff_inner == 0: continue
                            p = P * 2 + p_prime
                            q = Q * 2 + q_prime
                            w_term[p * 4 + q] += coeff_outer * coeff_inner
                            
            terms.append((u_term, v_term, w_term))
            
    if time.time() > deadline:
        # Fallback to schoolbook if something goes wrong, though unlikely
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Construct full U, V, W for 4x4x8
    # Two blocks of 4x4x4
    for block in range(2):
        col_offset = block * 49
        for t_idx in range(49):
            col = col_offset + t_idx
            u_term, v_term, w_term = terms[t_idx]
            
            # U is 16x16 local, maps to 16x1 global
            for i in range(16):
                U[i][col] = u_term[i]
                
            # V is 16x16 local (j, k'), maps to 16x32 global (j, k)
            # Local k' is 0..3. Global k is 4*block + k'
            for j in range(4):
                for k_prime in range(4):
                    local_idx = j * 4 + k_prime
                    global_idx = j * 8 + (4 * block + k_prime)
                    V[global_idx][col] = v_term[local_idx]
                    
            # W is 16x16 local (k', i), maps to 32x16 global (k, i)
            # Local k' is 0..3. Global k is 4*block + k'
            for k_prime in range(4):
                for i in range(4):
                    local_idx = k_prime * 4 + i
                    global_idx = (4 * block + k_prime) * 4 + i
                    W[global_idx][col] = w_term[local_idx]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
