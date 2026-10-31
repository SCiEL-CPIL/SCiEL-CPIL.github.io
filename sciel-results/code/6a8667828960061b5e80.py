import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4

    # Rank 49 decomposition using recursive Strassen.
    # 4x4 matrices are split into 2x2 blocks of 2x2 submatrices.
    # Outer Strassen (7 terms) on blocks, Inner Strassen (7 terms) on submatrices.
    # Total rank = 7 * 7 = 49.
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Index mapping helpers
    def a_idx(br, bc, lr, lc):
        return (br*2 + lr)*4 + (bc*2 + lc)
        
    def b_idx(br, bc, lr, lc):
        return (br*2 + lr)*4 + (bc*2 + lc)
        
    def w_idx(br, bc, ir, ic):
        # W index = col*4 + row
        # C row = br*2+ir, C col = bc*2+ic
        return (bc*2 + ic)*4 + (br*2 + ir)

    # Strassen 2x2 terms: (a_comb, b_comb, c_comb)
    # Each comb is a list of (local_row, local_col, coeff)
    strassen_2x2_terms = [
        # S1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # S2: (A10+A11)(B00) -> C10+=1, C11+=-1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # S3: (A00)(B01-B11) -> C01+=1, C11+=1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # S4: (A11)(B10-B00) -> C00+=1, C10+=1
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # S5: (A00+A01)(B11) -> C00+=-1, C01+=1
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # S6: (A10-A00)(B00+B01) -> C11+=1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # S7: (A01-A11)(B10+B11) -> C00+=1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Outer Strassen terms: (a_blocks, b_blocks, c_blocks)
    # Each block list is (block_row, block_col, coeff)
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
                    u_i = a_idx(br, bc, lr, lc)
                    U[u_i, col_idx] += coeff_a_outer * coeff_a_inner
            
            # Fill V
            for (br, bc, coeff_b_outer) in b_blocks:
                for (lr, lc, coeff_b_inner) in b_comb:
                    v_i = b_idx(br, bc, lr, lc)
                    V[v_i, col_idx] += coeff_b_outer * coeff_b_inner
            
            # Fill W
            for (cbr, cbc, coeff_c_outer) in c_blocks:
                for (ir, ic, coeff_c_inner) in c_comb:
                    w_i = w_idx(cbr, cbc, ir, ic)
                    W[w_i, col_idx] += coeff_c_outer * coeff_c_inner
            
            col_idx += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
