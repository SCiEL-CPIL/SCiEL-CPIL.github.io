DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # Order: (00, 01, 10, 11)
    # u: X (A block)
    # v: Y (B block)
    # w: Z (C block)
    strassen_products = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]), # M4
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 0, 1]), # M6
        ([0, -1, 0, 1], [0, 1, 0, 1], [1, 0, 0, 0]), # M7
    ]
    
    col = 0
    # Blocks: (i_start, j_start, k_start)
    # i_start is always 0 because n=2 and we split A into 2 rows?
    # No, A is 2x4. We split A into A1 (cols 0,1) and A2 (cols 2,3).
    # So i is always 0..1.
    # j splits into 0..1 and 2..3.
    # k splits into 0..1 and 2..3.
    
    blocks = [
        (0, 0, 0), # A1 B11 -> C1
        (0, 2, 0), # A2 B21 -> C1
        (0, 0, 2), # A1 B12 -> C2
        (0, 2, 2), # A2 B22 -> C2
    ]
    
    for i_s, j_s, k_s in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_products:
            # Update U
            for a in range(2):
                for b in range(2):
                    val = u_coeffs[b*2 + a]
                    if val != 0:
                        u_idx = (i_s + a) * m + (j_s + b)
                        U[u_idx][col] += val
            
            # Update V
            for b in range(2):
                for c in range(2):
                    val = v_coeffs[c*2 + b]
                    if val != 0:
                        v_idx = (j_s + b) * p + (k_s + c)
                        V[v_idx][col] += val
            
            # Update W
            for a in range(2):
                for c in range(2):
                    val = w_coeffs[c*2 + a]
                    if val != 0:
                        w_idx = (k_s + c) * n + (i_s + a)
                        W[w_idx][col] += val
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
