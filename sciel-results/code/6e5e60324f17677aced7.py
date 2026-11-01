import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We partition the p=8 columns into p/2 = 4 blocks of size 2.
    # For each block, we use Strassen's algorithm (rank 7).
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 4 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 16 x 28
    
    # Strassen linear forms for A (2x2)
    # A = [[a, b], [c, d]] -> indices 0,1,2,3
    # u1: a+d
    # u2: c+d
    # u3: a
    # u4: d
    # u5: a+b
    # u6: c+d
    # u7: a-d
    U_forms = [
        [1, 0, 0, 1], # 1
        [0, 0, 1, 1], # 2
        [1, 0, 0, 0], # 3
        [0, 0, 0, 1], # 4
        [1, 1, 0, 0], # 5
        [0, 0, 1, 1], # 6
        [1, 0, 0, -1]  # 7
    ]
    
    # Strassen linear forms for B block (2x2)
    # B = [[e, f], [g, h]]
    # In V, for a block starting at column k0 (0-indexed in B),
    # e is at V row: 0*p + k0 = k0
    # f is at V row: 0*p + (k0+1) = k0+1
    # g is at V row: 1*p + k0 = p + k0
    # h is at V row: 1*p + (k0+1) = p + k0 + 1
    # p=8.
    
    # W forms for C block (2x2)
    # C = [[c00, c01], [c10, c11]]
    # In W, for a block starting at column k0,
    # c00 is at W row: k0*n + 0 = 2*k0
    # c01 is at W row: (k0+1)*n + 0 = 2*k0 + 2
    # c10 is at W row: k0*n + 1 = 2*k0 + 1
    # c11 is at W row: (k0+1)*n + 1 = 2*k0 + 3
    
    # Coefficients for W based on Strassen reconstruction:
    # M1: c00+1, c11+1
    # M2: c10+1, c11-1
    # M3: c01+1, c11+1
    # M4: c00+1, c10+1
    # M5: c00-1, c01+1
    # M6: c11+1
    # M7: c00+1
    
    W_coeffs = [
        {0: 1, 3: 1},   # M1: c00 (idx 0), c11 (idx 3)
        {1: 1, 3: -1},  # M2: c10 (idx 1), c11 (idx 3)
        {2: 1, 3: 1},   # M3: c01 (idx 2), c11 (idx 3)
        {0: 1, 1: 1},   # M4: c00 (idx 0), c10 (idx 1)
        {0: -1, 2: 1},  # M5: c00 (idx 0), c01 (idx 2)
        {3: 1},         # M6: c11 (idx 3)
        {0: 1}          # M7: c00 (idx 0)
    ]
    
    num_blocks = p // 2 # 4 blocks
    
    for j in range(num_blocks):
        k0 = 2 * j # Start column index in B and C
        
        # V indices for this block
        v_e = k0
        v_f = k0 + 1
        v_g = p + k0
        v_h = p + k0 + 1
        
        # W indices for this block
        w_c00 = 2 * k0
        w_c01 = 2 * k0 + 2
        w_c10 = 2 * k0 + 1
        w_c11 = 2 * k0 + 3
        
        w_indices_map = [w_c00, w_c10, w_c01, w_c11] # 0:c00, 1:c10, 2:c01, 3:c11
        
        for t in range(7):
            col_idx = j * 7 + t
            
            # Fill U
            for i in range(4):
                U[i][col_idx] = U_forms[t][i]
                
            # Fill V
            V[v_e][col_idx] = 0
            V[v_f][col_idx] = 0
            V[v_g][col_idx] = 0
            V[v_h][col_idx] = 0
            
            if t == 0: # M1: e+h
                V[v_e][col_idx] = 1
                V[v_h][col_idx] = 1
            elif t == 1: # M2: e
                V[v_e][col_idx] = 1
            elif t == 2: # M3: f-e
                V[v_f][col_idx] = 1
                V[v_e][col_idx] = -1
            elif t == 3: # M4: h-g
                V[v_h][col_idx] = 1
                V[v_g][col_idx] = -1
            elif t == 4: # M5: h
                V[v_h][col_idx] = 1
            elif t == 5: # M6: g-h
                V[v_g][col_idx] = 1
                V[v_h][col_idx] = -1
            elif t == 6: # M7: e+f
                V[v_e][col_idx] = 1
                V[v_f][col_idx] = 1
                
            # Fill W
            for w_idx, val in W_coeffs[t].items():
                W[w_indices_map[w_idx]][col_idx] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
