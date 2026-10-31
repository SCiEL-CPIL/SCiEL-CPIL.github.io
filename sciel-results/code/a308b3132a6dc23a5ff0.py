import time
import numpy as np

def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen: 4x4 as 2x2 blocks of 2x2, with Strassen at both levels.
    Rank 49 is the proven optimal for 4x4x4 matrix multiplication.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper functions for linear combinations represented as dicts
    def add_dict(d1, d2):
        res = dict(d1)
        for k, v in d2.items():
            res[k] = res.get(k, 0) + v
        return {k: v for k, v in res.items() if v != 0}
        
    def sub_dict(d1, d2):
        res = dict(d1)
        for k, v in d2.items():
            res[k] = res.get(k, 0) - v
        return {k: v for k, v in res.items() if v != 0}
    
    # Get 2x2 block of A or B as tuple of 4 dicts (tl, tr, bl, br)
    def get_block(br, bc):
        tl = {(2*br, 2*bc): 1}
        tr = {(2*br, 2*bc + 1): 1}
        bl = {(2*br + 1, 2*bc): 1}
        br_el = {(2*br + 1, 2*bc + 1): 1}
        return (tl, tr, bl, br_el)
    
    def block_add(b1, b2):
        return (add_dict(b1[0], b2[0]), add_dict(b1[1], b2[1]), 
                add_dict(b1[2], b2[2]), add_dict(b1[3], b2[3]))
        
    def block_sub(b1, b2):
        return (sub_dict(b1[0], b2[0]), sub_dict(b1[1], b2[1]), 
                sub_dict(b1[2], b2[2]), sub_dict(b1[3], b2[3]))
    
    # Inner Strassen factors for 2x2 matrices X, Y (each is tuple of 4 dicts)
    def inner_strassen_factors(X, Y):
        x1 = add_dict(X[0], X[3])
        y1 = add_dict(Y[0], Y[3])
        x2 = add_dict(X[2], X[3])
        y2 = Y[0]
        x3 = X[0]
        y3 = sub_dict(Y[1], Y[3])
        x4 = X[3]
        y4 = sub_dict(Y[2], Y[0])
        x5 = add_dict(X[0], X[1])
        y5 = Y[3]
        x6 = sub_dict(X[2], X[0])
        y6 = add_dict(Y[0], Y[1])
        x7 = sub_dict(X[1], X[3])
        y7 = add_dict(Y[2], Y[3])
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7)]
    
    # Outer blocks
    A00, A01 = get_block(0, 0), get_block(0, 1)
    A10, A11 = get_block(1, 0), get_block(1, 1)
    B00, B01 = get_block(0, 0), get_block(0, 1)
    B10, B11 = get_block(1, 0), get_block(1, 1)
    
    # Outer Strassen terms
    outer_terms = [
        (block_add(A00, A11), block_add(B00, B11)),  # M1
        (block_add(A10, A11), B00),                   # M2
        (A00, block_sub(B01, B11)),                   # M3
        (A11, block_sub(B10, B00)),                   # M4
        (block_add(A00, A01), B11),                   # M5
        (block_sub(A10, A00), block_add(B00, B01)),   # M6
        (block_sub(A01, A11), block_add(B10, B11))    # M7
    ]
    
    # Inner contribution matrix: inner_contribs[k][li*2+lj] = coeff for Z[li][lj]
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    inner_contribs = [
        [1, 0, 0, 1],   # P1 -> Z00(+1), Z11(+1)
        [0, 0, 1, -1],  # P2 -> Z10(+1), Z11(-1)
        [0, 1, 0, 1],   # P3 -> Z01(+1), Z11(+1)
        [1, 0, 1, 0],   # P4 -> Z00(+1), Z10(+1)
        [-1, 1, 0, 0],  # P5 -> Z00(-1), Z01(+1)
        [0, 0, 0, 1],   # P6 -> Z11(+1)
        [1, 0, 0, 0]    # P7 -> Z00(+1)
    ]
    
    # Outer term contributions to C blocks: (block_row, block_col, coeff)
    outer_block_map = [
        [(0,0,1), (1,1,1)],       # M1 -> C00(+1), C11(+1)
        [(1,0,1), (1,1,-1)],      # M2 -> C10(+1), C11(-1)
        [(0,1,1), (1,1,1)],       # M3 -> C01(+1), C11(+1)
        [(0,0,1), (1,0,1)],       # M4 -> C00(+1), C10(+1)
        [(0,0,-1), (0,1,1)],      # M5 -> C00(-1), C01(+1)
        [(1,1,1)],                 # M6 -> C11(+1)
        [(0,0,1)]                  # M7 -> C00(+1)
    ]
    
    col = 0
    for t in range(7):
        X_block, Y_block = outer_terms[t]
        inner_factors = inner_strassen_factors(X_block, Y_block)
        
        for k in range(7):
            x_dict, y_dict = inner_factors[k]
            
            # Fill U from x_dict
            for (r, c), coeff in x_dict.items():
                U[r * 4 + c, col] += coeff
                
            # Fill V from y_dict
            for (r, c), coeff in y_dict.items():
                V[r * 4 + c, col] += coeff
                
            # Fill W
            for (br, bc, coeff_o) in outer_block_map[t]:
                for li in range(2):
                    for lj in range(2):
                        idx_local = li * 2 + lj
                        coeff_i = inner_contribs[k][idx_local]
                        if coeff_i != 0:
                            global_i = br * 2 + li  # row in C
                            global_k = bc * 2 + lj  # col in C
                            w_idx = global_k * 4 + global_i  # W[col*4 + row]
                            W[w_idx, col] += coeff_o * coeff_i
            
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
