import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4
    
    # Rank 49 decomposition using recursive Strassen.
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Index mapping helpers
    # A is 4x4, index = row * 4 + col
    # B is 4x4, index = row * 4 + col
    # C is 4x4, index = col * 4 + row (per prompt W[k*n+i])
    
    def a_idx(r, c):
        return r * 4 + c
        
    def b_idx(r, c):
        return r * 4 + c
        
    def w_idx(r, c):
        return c * 4 + r

    # Strassen 2x2 terms for local 2x2 blocks
    # Each term is (A_comb, B_comb, C_comb)
    # Comb is list of (row, col, coeff)
    strassen_2x2_terms = [
        # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2: (A10+A11)(B00) -> C10+=1, C11+=-1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3: (A00)(B01-B11) -> C01+=1, C11+=1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4: (A11)(B10-B00) -> C00+=1, C10+=1
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5: (A00+A01)(B11) -> C00+=-1, C01+=1
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6: (A10-A00)(B00+B01) -> C11+=1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7: (A01-A11)(B10+B11) -> C00+=1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Outer Strassen terms for 4x4 blocks (2x2 blocks of 2x2)
    outer_strassen_terms = [
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    col_idx = 0
    for outer_term in outer_strassen_terms:
        a_blocks, b_blocks, c_blocks = outer_term
        
        for inner_term in strassen_2x2_terms:
            a_comb, b_comb, c_comb = inner_term
            
            # Fill U
            for (br, bc, coeff_a_outer) in a_blocks:
                for (lr, lc, coeff_a_inner) in a_comb:
                    r = br * 2 + lr
                    c = bc * 2 + lc
                    u_i = a_idx(r, c)
                    U[u_i, col_idx] += coeff_a_outer * coeff_a_inner
            
            # Fill V
            for (br, bc, coeff_b_outer) in b_blocks:
                for (lr, lc, coeff_b_inner) in b_comb:
                    r = br * 2 + lr
                    c = bc * 2 + lc
                    v_i = b_idx(r, c)
                    V[v_i, col_idx] += coeff_b_outer * coeff_b_inner
            
            # Fill W
            for (cbr, cbc, coeff_c_outer) in c_blocks:
                for (ir, ic, coeff_c_inner) in c_comb:
                    r = cbr * 2 + ir
                    c = cbc * 2 + ic
                    w_i = w_idx(r, c)
                    W[w_i, col_idx] += coeff_c_outer * coeff_c_inner
            
            col_idx += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
