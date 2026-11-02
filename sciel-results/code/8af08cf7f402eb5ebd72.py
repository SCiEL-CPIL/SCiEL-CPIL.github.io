"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 56 (7 Strassen terms * 8 schoolbook terms per 2x2 block)
    rank = 56
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # Blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # alpha[t][b] is coeff of A_block_b in X_t
    # beta[t][c] is coeff of B_block_c in Y_t
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Block indices:
    # 0: 11, 1: 12, 2: 21, 3: 22
    
    alpha = [
        [1, 0, 0, 1], # M1: A11 + A22
        [0, 0, 1, 1], # M2: A21 + A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11 + A12
        [-1, 0, 1, 0],# M6: A21 - A11
        [0, 1, 0, -1] # M7: A12 - A22
    ]
    
    beta = [
        [1, 0, 0, 1], # M1: B11 + B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12 - B22
        [0, 0, 1, -1],# M4: B21 - B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11 + B12
        [0, 0, 1, 1]  # M7: B21 + B22
    ]
    
    # gamma[t][p][q] is coeff of M_t in C_pq
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # p is row block (0,1), q is col block (0,1)
    gamma = [
        # t=0 (M1)
        [[1, 0],
         [0, 1]],
        # t=1 (M2)
        [[0, 0],
         [1, -1]],
        # t=2 (M3)
        [[0, 1],
         [0, 1]],
        # t=3 (M4)
        [[1, 0],
         [1, 0]],
        # t=4 (M5)
        [[-1, 1],
         [0, 0]],
        # t=5 (M6)
        [[0, 0],
         [0, 1]],
        # t=6 (M7)
        [[1, 0],
         [0, 0]]
    ]
    
    col = 0
    for t in range(7):
        for u in range(2):
            for v in range(2):
                for w in range(2):
                    # This is one of the 8 terms in the 2x2 multiplication X_t * Y_t
                    # Term: X_t[u,v] * Y_t[v,w]
                    
                    # Calculate U contributions
                    for b in range(4):
                        a_coeff = alpha[t][b]
                        if a_coeff == 0: continue
                        
                        b_r = b // 2
                        b_c = b % 2
                        I = 2 * b_r + u
                        J = 2 * b_c + v
                        U[I * m + J][col] += a_coeff
                    
                    # Calculate V contributions
                    for c in range(4):
                        b_coeff = beta[t][c]
                        if b_coeff == 0: continue
                        
                        c_r = c // 2
                        c_c = c % 2
                        J_prime = 2 * c_r + v
                        K = 2 * c_c + w
                        V[J_prime * p + K][col] += b_coeff
                    
                    # Calculate W contributions
                    for p_blk in range(2):
                        for q_blk in range(2):
                            g_coeff = gamma[t][p_blk][q_blk]
                            if g_coeff == 0: continue
                            
                            I_out = 2 * p_blk + u
                            K_out = 2 * q_blk + w
                            # W index is k*n + i -> K_out * n + I_out
                            W[K_out * n + I_out][col] += g_coeff
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
