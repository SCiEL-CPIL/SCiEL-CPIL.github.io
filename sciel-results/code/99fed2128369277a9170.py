import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size
    
    deadline = time.time() + 145
    
    # Strassen decomposition at the 2x2 block level
    # 7 products
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # P1
        ([(2, 1), (3, 1)], [(0, 1)]),            # P2
        ([(0, 1)], [(1, 1), (3, -1)]),           # P3
        ([(3, 1)], [(2, 1), (0, -1)]),           # P4
        ([(0, 1), (1, 1)], [(3, 1)]),            # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),   # P7
    ]
    
    # Output reconstruction for Strassen:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    output_contrib = [
        [(0, 1), (3, 1)],     # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],    # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],     # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],     # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],    # P5 -> C00(-1), C01(+1)
        [(3, 1)],             # P6 -> C11(+1)
        [(0, 1)],             # P7 -> C00(+1)
    ]
    
    # Schoolbook decomposition for 3x3 matrix multiplication
    # Each product P_k is a 3x3 matrix product.
    # We decompose each 3x3 product into 27 scalar products.
    # Total rank = 7 * 27 = 189.
    
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        
        # Iterate over all 3x3 scalar products for this block product
        for i in range(3):      # row of A block (and C block)
            for k in range(3):  # col of A block (and row of B block)
                for j in range(3): # col of B block (and C block)
                    
                    # A contribution
                    for (blk_a, sa) in a_comb:
                        row_a = (blk_a // 2) * bs + i
                        col_a = (blk_a % 2) * bs + k
                        u_idx = row_a * m + col_a
                        U[u_idx][col] += sa
                    
                    # B contribution
                    for (blk_b, sb) in b_comb:
                        row_b = (blk_b // 2) * bs + k
                        col_b = (blk_b % 2) * bs + j
                        v_idx = row_b * p + col_b
                        V[v_idx][col] += sb
                    
                    # C contribution
                    # The product is added to the corresponding C block
                    for (blk_c, sc) in output_contrib[term_idx]:
                        row_c = (blk_c // 2) * bs + i
                        col_c = (blk_c % 2) * bs + j
                        w_idx = col_c * n + row_c
                        W[w_idx][col] += sc
                        
                    col += 1
        
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
