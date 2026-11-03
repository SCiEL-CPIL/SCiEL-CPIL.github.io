import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, term_u, term_v, term_w):
        for idx, coeff in term_u:
            U[idx][col] += coeff
        for idx, coeff in term_v:
            V[idx][col] += coeff
        for idx, coeff in term_w:
            W[idx][col] += coeff

    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(1, 1), (3, 1)], [(0, 1)]),          # M2
        ([(0, 1)], [(1, 1), (3, -1)]),         # M3
        ([(3, 1)], [(2, 1), (0, -1)]),         # M4
        ([(0, 1), (1, 1)], [(3, 1)]),          # M5
        ([(1, 1), (0, -1)], [(0, 1), (1, 1)]), # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])  # M7
    ]
    
    c_block_contribs = [
        [(0, 1), (3, 1)], # M1
        [(2, 1), (3, -1)], # M2
        [(1, 1), (3, 1)], # M3
        [(0, 1), (2, 1)], # M4
        [(0, -1), (1, 1)], # M5
        [(3, 1)],           # M6
        [(0, 1)]            # M7
    ]
    
    col_idx = 0
    for t_idx, (X_blocks, Y_blocks) in enumerate(strassen_terms):
        for i in range(3):
            for k in range(3):
                u_terms = []
                for b_idx, coeff in X_blocks:
                    row = 3 * (b_idx // 2) + i
                    col = 3 * (b_idx % 2) + k
                    u_terms.append((row * 6 + col, coeff))
                
                for j in range(3):
                    v_terms = []
                    for b_idx, coeff in Y_blocks:
                        row = 3 * (b_idx // 2) + k
                        col = 3 * (b_idx % 2) + j
                        v_terms.append((row * 6 + col, coeff))
                    
                    for c_b_idx, c_coeff in c_block_contribs[t_idx]:
                        w_row = 3 * (c_b_idx // 2) + i
                        w_col = 3 * (c_b_idx % 2) + j
                        w_terms = [(w_row * 6 + w_col, c_coeff)]
                        
                        add_term(col_idx, u_terms, v_terms, w_terms)
                        col_idx += 1
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
