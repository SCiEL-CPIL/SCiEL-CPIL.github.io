import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for 2x2x2
    # M1: A(a+d), B(e+h)
    # M2: A(c+d), B(e)
    # M3: A(a), B(f-h)
    # M4: A(d), B(g-c)
    # M5: A(a+b), B(h)
    # M6: A(c-a), B(e+f)
    # M7: A(b-d), B(g+h)
    
    # Output:
    # C11: M1 + M4 - M5 + M7
    # C12: M3 + M5
    # C21: M2 + M4
    # C22: M1 - M2 + M3 + M6
    
    # We define the contributions for each of the 7 Strassen steps
    # Each step is a tuple (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping local A index to coeff
    # v_coeffs: dict mapping local B index to coeff
    # w_coeffs: dict mapping local C index to coeff
    
    # Local indices:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11 (where 00 is row 0 col 0 of the 2x2 block)
    # C: 00, 01, 10, 11
    
    strassen_steps = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
    ]
    
    # Process Block 1: A cols 0,1; B rows 0,1
    for pair_idx in range(4):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        
        # Mapping local indices to global indices
        # A local (i, j_local) -> global (i, j_global)
        # j_global = j_local (since block 1 uses j=0,1)
        # B local (j_local, k_local) -> global (j_global, k_global)
        # j_global = j_local (since block 1 uses j=0,1)
        # k_global = k1 + k_local (k_local is 0 or 1)
        # C local (i, k_local) -> global (i, k_global)
        
        for step in strassen_steps:
            u_coeffs, v_coeffs, w_coeffs = step
            
            # U
            for (li, lj), coeff in u_coeffs.items():
                global_i = li
                global_j = lj
                U[global_i * m + global_j][col] += coeff
                
            # V
            for (lj, lk), coeff in v_coeffs.items():
                global_j = lj
                global_k = k1 + lk
                V[global_j * p + global_k][col] += coeff
                
            # W
            for (li, lk), coeff in w_coeffs.items():
                global_i = li
                global_k = k1 + lk
                # W index is k*n + i
                W[global_k * n + global_i][col] += coeff
                
            col += 1
            
    # Process Block 2: A cols 2,3; B rows 2,3
    for pair_idx in range(4):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        
        # Mapping local indices to global indices
        # A local (i, j_local) -> global (i, j_global)
        # j_global = 2 + j_local
        # B local (j_local, k_local) -> global (j_global, k_global)
        # j_global = 2 + j_local
        # k_global = k1 + k_local
        # C local (i, k_local) -> global (i, k_global)
        
        for step in strassen_steps:
            u_coeffs, v_coeffs, w_coeffs = step
            
            # U
            for (li, lj), coeff in u_coeffs.items():
                global_i = li
                global_j = 2 + lj
                U[global_i * m + global_j][col] += coeff
                
            # V
            for (lj, lk), coeff in v_coeffs.items():
                global_j = 2 + lj
                global_k = k1 + lk
                V[global_j * p + global_k][col] += coeff
                
            # W
            for (li, lk), coeff in w_coeffs.items():
                global_i = li
                global_k = k1 + lk
                # W index is k*n + i
                W[global_k * n + global_i][col] += coeff
                
            col += 1

    return {"U": U, "V": V, "W": W}
