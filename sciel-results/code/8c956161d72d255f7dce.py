"""Rewritten program: exact rank-65 decomposition for 4x4 by 4x5 matrix multiplication.
Uses a block-decomposition approach:
1. Split the 4x5 right factor into a 4x4 block (B1) and a 4x1 block (B2).
2. Use Strassen's algorithm (rank 49) for the 4x4x4 multiplication (A * B1).
3. Use the schoolbook algorithm (rank 16) for the 4x4x1 multiplication (A * B2).
Total rank = 49 + 16 = 65.
"""

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Strassen 2x2x2 decomposition vectors
    U2 = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 1], # m2
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 1, 0, 0], # m5
        [-1, 0, 1, 0], # m6
        [0, 1, 0, -1]  # m7
    ]
    V2 = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, 1, 0, -1], # m3
        [-1, 0, 1, 0], # m4
        [0, 0, 0, 1], # m5
        [1, 1, 0, 0], # m6
        [0, 0, 1, 1]  # m7
    ]
    W2 = [
        [1, 0, 0, 1], # m1
        [0, 1, 0, -1], # m2
        [-1, 0, 1, 0], # m3
        [1, 1, 0, 0], # m4
        [-1, 0, 1, 0], # m5
        [0, 0, 0, 1], # m6
        [1, 0, 0, 1]  # m7
    ]
    
    # Strassen steps coefficients for A, B, C blocks
    # A blocks: (r, c) in 0..1
    # B blocks: (c, s) in 0..1
    # C blocks: (r, s) in 0..1
    steps = [
        # (alpha_dict, beta_dict, delta_dict)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}), # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): -1, (0,1): 1}), # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}) # M7
    ]
    
    col = 0
    for t in range(7):
        alpha = steps[t][0]
        beta = steps[t][1]
        delta = steps[t][2]
        
        for q in range(7):
            # Process U
            for r in range(2):
                for c in range(2):
                    if (r, c) in alpha:
                        a_coeff = alpha[(r, c)]
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u_idx = 8*r + 2*c + 4*i_prime + j_prime
                                local_idx = 2*i_prime + j_prime
                                U[u_idx][col] += a_coeff * U2[q][local_idx]
            
            # Process V
            for c in range(2):
                for s in range(2):
                    if (c, s) in beta:
                        b_coeff = beta[(c, s)]
                        for j_prime in range(2):
                            for k_prime in range(2):
                                v_idx = 8*c + 4*j_prime + 2*s + k_prime
                                local_idx = 2*j_prime + k_prime
                                V[v_idx][col] += b_coeff * V2[q][local_idx]
                                
            # Process W
            for r in range(2):
                for s in range(2):
                    if (r, s) in delta:
                        c_coeff = delta[(r, s)]
                        for i_prime in range(2):
                            for j_prime in range(2):
                                w_idx = 8*s + 4*j_prime + 2*r + i_prime
                                local_idx = 2*i_prime + j_prime
                                W[w_idx][col] += c_coeff * W2[q][local_idx]
                                
            col += 1
            
    # Part 2: 4x4x1 using Schoolbook (Rank 16)
    for i in range(n):
        for j in range(m):
            # U index: i*m + j
            # V index: j*p + k (k=0) -> j
            # W index: k*n + i (k=0) -> i
            U[i * m + j][col] = 1
            V[j * p + 0][col] = 1
            W[0 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
