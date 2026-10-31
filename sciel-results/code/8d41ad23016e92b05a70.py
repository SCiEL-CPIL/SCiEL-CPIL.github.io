import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Base Strassen factors for 2x2
    # u, v, w are lists of 4 coefficients [00, 01, 10, 11]
    base_factors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Block level factors k=1..7
    # X, Y are defined by coefficients on A_blocks and B_blocks
    # A_blocks: 00, 01, 10, 11
    # B_blocks: 00, 01, 10, 11
    # X[r, s] = sum coeff * A_block
    # We represent X as a 2x2 matrix of coefficients for each A_block
    # Or simpler: for each local (r,s), X[r,s] is a linear form on A.
    
    # Let's define the linear forms for X and Y for each k
    # k=1: X = A00 + A11, Y = B00 + B11
    # k=2: X = A10 + A11, Y = B00
    # k=3: X = A00, Y = B01 - B11
    # k=4: X = A11, Y = B10 - B00
    # k=5: X = A00 + A01, Y = B11
    # k=6: X = A10 - A00, Y = B00 + B01
    # k=7: X = A01 - A11, Y = B10 + B11
    
    # We need to map A_block (I, J) to global indices
    # A[I, J, r, s] -> i = 2I+r, j = 2J+s -> idx = i*4+j
    
    # For each k, we define X_coeffs[r][s] as a dict { (I,J): coeff }
    # And Y_coeffs[r][s] as a dict { (J,K): coeff }
    
    def get_x_coeffs(k):
        if k == 1:
            return { (0,0): 1, (1,1): 1 } # Same for all r,s
        elif k == 2:
            return { (1,0): 1, (1,1): 1 }
        elif k == 3:
            return { (0,0): 1 }
        elif k == 4:
            return { (1,1): 1 }
        elif k == 5:
            return { (0,0): 1, (0,1): 1 }
        elif k == 6:
            return { (1,0): 1, (0,0): -1 }
        elif k == 7:
            return { (0,1): 1, (1,1): -1 }
            
    def get_y_coeffs(k):
        if k == 1:
            return { (0,0): 1, (1,1): 1 }
        elif k == 2:
            return { (0,0): 1 }
        elif k == 3:
            return { (0,1): 1, (1,1): -1 }
        elif k == 4:
            return { (1,0): 1, (0,0): -1 }
        elif k == 5:
            return { (1,1): 1 }
        elif k == 6:
            return { (0,0): 1, (0,1): 1 }
        elif k == 7:
            return { (1,0): 1, (1,1): 1 }
            
    # Alpha coefficients for C blocks
    # C[I, K] = sum alpha[I, K, k] M_k
    alphas = {}
    alphas[(0,0)] = {1: 1, 4: 1, 5: -1, 7: 1}
    alphas[(0,1)] = {3: 1, 5: 1}
    alphas[(1,0)] = {2: 1, 4: 1}
    alphas[(1,1)] = {1: 1, 2: -1, 3: 1, 6: 1}
    
    col = 0
    for k in range(1, 8):
        x_coeffs = get_x_coeffs(k)
        y_coeffs = get_y_coeffs(k)
        
        for c in range(7):
            u_base, v_base, w_base = base_factors[c]
            
            # Construct U linear form
            # U[idx_A] += sum_{r,s} u_base[r*2+s] * X[r,s]
            # X[r,s] is a linear form on A blocks.
            # Since x_coeffs is the same for all r,s in our specific Strassen definitions?
            # Check:
            # k=1: X = A00+A11. X[r,s] = A00[r,s] + A11[r,s].
            # Yes, the block addition is element-wise.
            # So X[r,s] uses the same set of blocks with same coeffs for all r,s.
            
            for r in range(2):
                for s in range(2):
                    u_val = u_base[r*2+s]
                    if u_val == 0: continue
                    for (I, J), coeff in x_coeffs.items():
                        i = 2*I + r
                        j = 2*J + s
                        idx_A = i*4 + j
                        U[idx_A][col] += u_val * coeff
                        
            # Construct V linear form
            for r in range(2):
                for s in range(2):
                    v_val = v_base[r*2+s]
                    if v_val == 0: continue
                    for (J, K), coeff in y_coeffs.items():
                        j = 2*J + r
                        k_idx = 2*K + s
                        idx_B = j*4 + k_idx
                        V[idx_B][col] += v_val * coeff
                        
            # Construct W linear form
            # W[idx_W] = alpha[I, K, k] * w_base[r*2+s]
            # Note: w_base is for the output of the 2x2 product M_k.
            # M_k[r,s] contributes to C[I, K, r, s] with coeff alpha.
            # C[I, K, r, s] has global index i = 2I+r, k_global = 2K+s.
            # W row index is k_global * 4 + i.
            
            for I in range(2):
                for K in range(2):
                    alpha = alphas.get((I, K), {}).get(k, 0)
                    if alpha == 0: continue
                    for r in range(2):
                        for s in range(2):
                            w_val = w_base[r*2+s]
                            if w_val == 0: continue
                            i = 2*I + r
                            k_global = 2*K + s
                            idx_W = k_global * 4 + i
                            W[idx_W][col] += alpha * w_val
                            
            col += 1
            
    return {"U": U, "V": V, "W": W}
