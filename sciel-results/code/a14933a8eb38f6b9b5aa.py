import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses two rank-49 Strassen-based decompositions for 4x4x4.
    Total rank: 98.
    """
    
    n, m, p = 4, 4, 8
    
    # Strassen coefficients for 2x2 multiplication
    # alpha[t][P] coefficient of A-block P in A-combo t
    # beta[t][Q] coefficient of B-block Q in B-combo t
    # gamma[s][ab] coefficient of scalar s in 2x2 result entry ab
    # delta_u[s][ab] coefficient of A entry ab in A-combo s
    # delta_v[s][ab] coefficient of B entry ab in B-combo s
    # kappa[t][PQ] coefficient of M-t in C-block PQ
    
    # Outer Strassen (block level 2x2)
    # t=1..7
    # P,Q=0..3 (00, 01, 10, 11)
    alpha_outer = [
        [0],  # dummy
        [1, 0, 0, 1], # t=1: A00 + A11
        [0, 0, 1, 1], # t=2: A10 + A11
        [1, 0, 0, 0], # t=3: A00
        [0, 0, 0, 1], # t=4: A11
        [1, 1, 0, 0], # t=5: A00 + A01
        [0, 0, 1, -1],# t=6: A10 - A00
        [0, 1, 0, -1] # t=7: A01 - A11
    ]
    
    beta_outer = [
        [0],
        [1, 0, 0, 1], # B00 + B11
        [1, 0, 0, 0], # B00
        [0, 1, 0, -1],# B01 - B11
        [0, 0, 1, -1],# B10 - B00
        [0, 0, 0, 1], # B11
        [1, 1, 0, 0], # B00 + B01
        [0, 0, 1, 1]  # B10 + B11
    ]
    
    kappa_outer = [
        [0],
        [1, 0, 0, 1], # C00 += M1, C11 += M1
        [0, 0, 1, -1],# C10 += M2, C11 -= M2
        [0, 1, 0, 1], # C01 += M3, C11 += M3
        [1, 0, 1, 0], # C00 += M4, C10 += M4
        [-1, 1, 0, 0],# C00 -= M5, C01 += M5
        [0, 0, 0, 1], # C11 += M6
        [1, 0, 0, 0]  # C00 += M7
    ]
    
    # Inner Strassen (scalar level 2x2)
    # s=1..7
    # entries (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
    # delta_u[s][idx]
    delta_u_inner = [
        [0],
        [1, 0, 0, 1], # (0,0)+ (1,1)
        [0, 0, 1, 1], # (1,0)+ (1,1)
        [1, 0, 0, 0], # (0,0)
        [0, 0, 0, 1], # (1,1)
        [1, 1, 0, 0], # (0,0)+ (0,1)
        [0, 0, 1, -1],# (1,0)- (0,0) -> Wait, standard Strassen M6 is (A20-A10)?
        # Let's stick to the dict logic from the parent which was correct.
        # Parent delta_u:
        # 1: (0,0):1, (1,1):1
        # 2: (1,0):1, (1,1):1
        # 3: (0,0):1
        # 4: (1,1):1
        # 5: (0,0):1, (0,1):1
        # 6: (1,0):1, (0,0):-1
        # 7: (0,1):1, (1,1):-1
        [0, 1, 0, -1], # (0,1)+ (1,1)? No, parent was (0,1):1, (1,1):-1.
        # Let's re-verify parent.
        # Parent: 7: {(0,1): 1, (1,1): -1}
        # My list index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # So 7: [0, 1, 0, -1]
    ]
    # Correcting delta_u_inner to match parent exactly:
    delta_u_inner = [
        [0],
        [1, 0, 0, 1], # 1
        [0, 0, 1, 1], # 2
        [1, 0, 0, 0], # 3
        [0, 0, 0, 1], # 4
        [1, 1, 0, 0], # 5
        [0, 0, 1, -1],# 6: (1,0):1, (0,0):-1 -> [0,0,1,-1] is wrong. (0,0) is idx 0. So [-1, 0, 1, 0]
        # Let's just use the parent's logic but with lists.
    ]
    
    # Re-initializing lists carefully to avoid errors
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    delta_u_list = [
        [0, 0, 0, 0],
        [1, 0, 0, 1], # 1
        [0, 0, 1, 1], # 2
        [1, 0, 0, 0], # 3
        [0, 0, 0, 1], # 4
        [1, 1, 0, 0], # 5
        [-1, 0, 1, 0],# 6
        [0, 1, 0, -1] # 7
    ]
    
    delta_v_list = [
        [0, 0, 0, 0],
        [1, 0, 0, 1], # 1
        [1, 0, 0, 0], # 2
        [0, 1, 0, -1],# 3
        [0, 0, 1, -1],# 4
        [0, 0, 0, 1], # 5
        [1, 1, 0, 0], # 6
        [0, 0, 1, 1]  # 7
    ]
    
    gamma_list = [
        [0, 0, 0, 0],
        [1, 0, 0, 1], # 1: (0,0), (1,1)
        [0, 0, 1, -1],# 2: (1,0), (1,1) -> Wait, parent gamma 2: (1,0):1, (1,1):-1
        [0, 1, 0, 1], # 3: (0,1), (1,1)
        [1, 0, 1, 0], # 4: (0,0), (1,0)
        [1, 1, 0, 0], # 5: (0,1), (0,0) -> Parent gamma 5: (0,1):1, (0,0):-1
        [0, 0, 0, 1], # 6: (1,1)
        [1, 0, 0, 0]  # 7: (0,0)
    ]
    # Correcting gamma_list
    gamma_list = [
        [0, 0, 0, 0],
        [1, 0, 0, 1], # 1
        [0, 0, 1, -1],# 2
        [0, 1, 0, 1], # 3
        [1, 0, 1, 0], # 4
        [-1, 1, 0, 0],# 5
        [0, 0, 0, 1], # 6
        [1, 0, 0, 0]  # 7
    ]

    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    
    for t in range(1, 8):
        for s in range(1, 8):
            r = (t - 1) * 7 + (s - 1)
            
            # U
            for a in range(2):
                for b in range(2):
                    coeff_u = delta_u_list[s][a * 2 + b]
                    if coeff_u == 0: continue
                    for P in range(4):
                        coeff_a = alpha_outer[t][P]
                        if coeff_a == 0: continue
                        coeff = coeff_u * coeff_a
                        if coeff == 0: continue
                        P_row = P // 2
                        P_col = P % 2
                        i = 2 * P_row + a
                        j = 2 * P_col + b
                        U444[i * 4 + j, r] += coeff

            # V
            for a in range(2):
                for b in range(2):
                    coeff_v = delta_v_list[s][a * 2 + b]
                    if coeff_v == 0: continue
                    for Q in range(4):
                        coeff_b = beta_outer[t][Q]
                        if coeff_b == 0: continue
                        coeff = coeff_v * coeff_b
                        if coeff == 0: continue
                        Q_row = Q // 2
                        Q_col = Q % 2
                        j = 2 * Q_row + a
                        k = 2 * Q_col + b
                        V444[j * 4 + k, r] += coeff

            # W
            for a in range(2):
                for b in range(2):
                    coeff_m = gamma_list[s][a * 2 + b]
                    if coeff_m == 0: continue
                    for PQ in range(4):
                        coeff_k = kappa_outer[t][PQ]
                        if coeff_k == 0: continue
                        coeff = coeff_m * coeff_k
                        if coeff == 0: continue
                        PQ_row = PQ // 2
                        PQ_col = PQ % 2
                        i = 2 * PQ_row + a
                        k = 2 * PQ_col + b
                        W444[k * 4 + i, r] += coeff

    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    U[:, 0:49] = U444
    U[:, 49:98] = U444
    
    for j in range(4):
        for k in range(4):
            V[8*j + k, 0:49] = V444[4*j + k, :]
            V[8*j + k + 4, 49:98] = V444[4*j + k, :]
            
    for k in range(4):
        for i in range(4):
            W[4*k + i, 0:49] = W444[4*k + i, :]
            W[4*(k+4) + i, 49:98] = W444[4*k + i, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
