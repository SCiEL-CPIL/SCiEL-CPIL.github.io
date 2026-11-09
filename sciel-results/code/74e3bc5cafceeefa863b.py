import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4 * 4x4 using Strassen (Rank 49)
    # Part 2: 4x4 * 4x3 using Trivial (Rank 48)
    total_rank = 49 + 48
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Part 1: Strassen for 4x4x4 ---
    # Strassen coefficients for 2x2
    # U_s: 7x4, V_s: 7x4, W_s: 7x4
    # Indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    # W index for 2x2 (n=2): k*2+i
    # C_00: 0, C_01: 2, C_10: 1, C_11: 3
    
    U_s = np.array([
        [1, 0, 0, 1], # M1
        [0, 1, 0, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 0, 1, 0], # M5
        [-1, 1, 0, 0],# M6
        [0, -1, 1, 0] # M7
    ])
    
    V_s = np.array([
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ])
    
    # W_s maps scalar products to C elements
    # C_00 (idx 0): M1
    # C_01 (idx 2): M3, M5, M7
    # C_10 (idx 1): M2, M4, M6
    # C_11 (idx 3): M1, M2, M5
    W_s = np.zeros((7, 4))
    W_s[0, 0] = 1; W_s[0, 3] = 1 # M1 -> C00, C11
    W_s[1, 1] = 1; W_s[1, 3] = 1 # M2 -> C10, C11
    W_s[2, 2] = 1                 # M3 -> C01
    W_s[3, 1] = 1                 # M4 -> C10
    W_s[4, 2] = 1; W_s[4, 3] = 1 # M5 -> C01, C11
    W_s[5, 1] = 1                 # M6 -> C10
    W_s[6, 2] = 1                 # M7 -> C01
    
    # Block Strassen coefficients for 4x4
    # Blocks: 00, 01, 10, 11 -> indices 0, 1, 2, 3 in block space
    # U_b: 7x4, V_b: 7x4, W_b: 7x4
    U_b = np.array([
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ])
    
    V_b = np.array([
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ])
    
    # W_b maps block products to C blocks
    # C_00 (block 0): M1
    # C_01 (block 1): M3, M5, M7
    # C_10 (block 2): M2, M4, M6
    # C_11 (block 3): M1, M2, M5
    W_b = np.zeros((7, 4))
    W_b[0, 0] = 1; W_b[0, 3] = 1
    W_b[1, 2] = 1; W_b[1, 3] = 1
    W_b[2, 1] = 1
    W_b[3, 2] = 1
    W_b[4, 1] = 1; W_b[4, 3] = 1
    W_b[5, 2] = 1
    W_b[6, 1] = 1
    
    col = 0
    for kb in range(7):
        for ls in range(7):
            # U vector: sum over block I,J and local a,b
            # A index: (2*I + a) * m + (2*J + b)
            # U_val = U_b[kb, I*2+J] * U_s[ls, a*2+b]
            for I in range(2):
                for J in range(2):
                    ub_val = U_b[kb, I * 2 + J]
                    if ub_val == 0: continue
                    for a in range(2):
                        for b in range(2):
                            us_val = U_s[ls, a * 2 + b]
                            if us_val == 0: continue
                            i = 2 * I + a
                            j = 2 * J + b
                            U[i * m + j, col] += ub_val * us_val
            
            # V vector: sum over block I,J and local c,d
            # B index: (2*I + c) * p + (2*J + d)  -- Wait, B is 4x4 here.
            # In the 4x4x4 subproblem, B is 4x4.
            # But in the full problem, B is 4x7.
            # This subproblem uses the first 4 columns of B.
            # So B index: (2*I + c) * p + (2*J + d) is WRONG.
            # B is m x p. m=4, p=7.
            # The subproblem B is 4x4.
            # So B index should be (2*I + c) * p + (2*J + d) ?
            # No, B is stored as m x p.
            # Row j, Col k. Index j * p + k.
            # In 4x4 subproblem, j in 0..3, k in 0..3.
            # So index is j * p + k.
            # j = 2*I + c, k = 2*J + d.
            for I in range(2):
                for J in range(2):
                    vb_val = V_b[kb, I * 2 + J]
                    if vb_val == 0: continue
                    for c in range(2):
                        for d in range(2):
                            vs_val = V_s[ls, c * 2 + d]
                            if vs_val == 0: continue
                            j = 2 * I + c
                            k = 2 * J + d
                            V[j * p + k, col] += vb_val * vs_val
            
            # W vector: sum over block K,L and local e,f
            # C index: (2*K + e) * n + (2*L + f) ?
            # W index convention: k * n + i.
            # C is n x p. n=4, p=7.
            # Subproblem C is 4x4.
            # i in 0..3, k in 0..3.
            # Index: k * n + i.
            # k = 2*K + f, i = 2*L + e.
            for K in range(2):
                for L in range(2):
                    wb_val = W_b[kb, K * 2 + L]
                    if wb_val == 0: continue
                    for e in range(2):
                        for f in range(2):
                            ws_val = W_s[ls, e * 2 + f]
                            if ws_val == 0: continue
                            i = 2 * L + e
                            k = 2 * K + f
                            W[k * n + i, col] += wb_val * ws_val
            
            col += 1
    
    # --- Part 2: Trivial for 4x4x3 ---
    # i in 0..3, j in 0..3, k in 4..6 (last 3 cols)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
