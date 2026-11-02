import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition using Strassen for the 2x2 part
    # Split m=3 into 1 + 2
    # Part 1: 2x1x10 (rank 20)
    # Part 2: 2x2x10 (rank 35, using 5 blocks of 2x2x2 Strassen)
    
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # A indices: i*3 + 0
    # B indices: 0*10 + k
    # C indices: k*2 + i
    for i in range(n):
        for k in range(p):
            U[i * m][col] = 1
            V[k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A[:, 1:3] * B[1:3, :]
    # A indices: i*3 + (1 + j_loc)
    # B indices: (1 + j_loc)*10 + k
    # C indices: k*2 + i
    
    # Split k into 5 groups of 2
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # Strassen 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Helper to add a term
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                U[idx][col] += val
            for idx, val in v_coeffs.items():
                V[idx][col] += val
            for idx, val in w_coeffs.items():
                W[idx][col] += val
            col += 1
        
        # Map local A indices (i, j_loc) to global U index
        def u_idx(i, j_loc):
            return i * m + (1 + j_loc)
            
        # Map local B indices (j_loc, k_loc) to global V index
        def v_idx(j_loc, k_loc):
            return (1 + j_loc) * p + (k0 + k_loc)
            
        # Map local C indices (k_loc, i) to global W index
        def w_idx(k_loc, i):
            return (k0 + k_loc) * n + i

        # M1
        add_term(
            {u_idx(0,0): 1, u_idx(1,1): 1},
            {v_idx(0,0): 1, v_idx(1,1): 1},
            {w_idx(0,0): 1, w_idx(0,1): 1, w_idx(1,0): 1, w_idx(1,1): -1} # C00 += M1, C01 += 0, C10 += 0, C11 -= M1? 
            # Wait, C = M1 + M4 - M5 + M7 (for C00)
            # C = M3 + M5 (for C01)
            # C = M2 + M4 (for C10)
            # C = M1 - M2 + M3 + M6 (for C11)
            # So for M1:
            # W[0,0] += 1
            # W[1,1] += 1
        )
        # Correction: W coefficients must match the C contribution.
        # C00 gets M1. C11 gets M1.
        # So W[w_idx(0,0)] = 1, W[w_idx(1,1)] = 1.
        
        # Let's restart the W logic for Strassen terms carefully.
        
        # M1
        add_term(
            {u_idx(0,0): 1, u_idx(1,1): 1},
            {v_idx(0,0): 1, v_idx(1,1): 1},
            {w_idx(0,0): 1, w_idx(1,1): 1}
        )
        
        # M2
        add_term(
            {u_idx(1,0): 1, u_idx(1,1): 1},
            {v_idx(0,0): 1},
            {w_idx(1,0): 1, w_idx(1,1): -1}
        )
        
        # M3
        add_term(
            {u_idx(0,0): 1},
            {v_idx(1,0): 1, v_idx(1,1): -1},
            {w_idx(0,1): 1, w_idx(1,1): 1}
        )
        
        # M4
        add_term(
            {u_idx(1,1): 1},
            {v_idx(0,1): 1, v_idx(0,0): -1},
            {w_idx(0,0): 1, w_idx(1,0): 1}
        )
        
        # M5
        add_term(
            {u_idx(0,0): 1, u_idx(0,1): 1},
            {v_idx(1,1): 1},
            {w_idx(0,0): -1, w_idx(0,1): 1}
        )
        
        # M6
        add_term(
            {u_idx(1,0): 1, u_idx(0,0): -1},
            {v_idx(0,0): 1, v_idx(0,1): 1},
            {w_idx(1,1): 1}
        )
        
        # M7
        add_term(
            {u_idx(0,1): 1, u_idx(1,1): -1},
            {v_idx(1,0): 1, v_idx(1,1): 1},
            {w_idx(0,0): 1, w_idx(1,1): 1}
        )

    return {"U": U, "V": V, "W": W}
