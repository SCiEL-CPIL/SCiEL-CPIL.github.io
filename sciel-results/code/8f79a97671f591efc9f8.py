"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size is 2, number of blocks is 3
    block_size = 2
    num_blocks = n // block_size  # 3
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-d)(f+h), M7 = (b-c)(e+g)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # Define the 7 Strassen terms as (coefficients in A block, coefficients in B block)
    # Each entry is (row, col, coeff)
    strassen_terms = [
        # M1 = (a+d)(e+h)
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (0,1,1), (1,0,-1), (1,1,-1)],  # Wait, need to be careful
        },
    ]
    
    # Let me redefine more carefully. Strassen's algorithm:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-d)*(f+h)
    # M7 = (b-c)*(e+g)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For each of the 7 products, define:
    # - Which elements of A block (u,v) are used with what coefficients
    # - Which elements of B block (v,w) are used with what coefficients
    # - Which elements of C block (u,w) receive what coefficients
    
    strassen_products = []
    
    # Product 1: M1 = (a+d)(e+h)
    # A: a(0,0) + d(1,1), B: e(0,0) + h(1,1)
    # C: C[0,0] += 1, C[1,1] += 1
    strassen_products.append({
        'A': [(0,0,1), (1,1,1)],
        'B': [(0,0,1), (1,1,1)],
        'C': [(0,0,1), (1,1,1)]
    })
    
    # Product 2: M2 = (c+d)*e
    # A: c(1,0) + d(1,1), B: e(0,0)
    # C: C[1,0] += 1, C[1,1] += -1
    strassen_products.append({
        'A': [(1,0,1), (1,1,1)],
        'B': [(0,0,1)],
        'C': [(1,0,1), (1,1,-1)]
    })
    
    # Product 3: M3 = a*(f-h)
    # A: a(0,0), B: f(0,1) - h(1,1)
    # C: C[0,1] += 1, C[1,1] += 1
    strassen_products.append({
        'A': [(0,0,1)],
        'B': [(0,1,1), (1,1,-1)],
        'C': [(0,1,1), (1,1,1)]
    })
    
    # Product 4: M4 = d*(g-e)
    # A: d(1,1), B: g(1,0) - e(0,0)
    # C: C[0,0] += 1, C[1,0] += 1
    strassen_products.append({
        'A': [(1,1,1)],
        'B': [(1,0,1), (0,0,-1)],
        'C': [(0,0,1), (1,0,1)]
    })
    
    # Product 5: M5 = (a+b)*h
    # A: a(0,0) + b(0,1), B: h(1,1)
    # C: C[0,0] += -1, C[0,1] += 1
    strassen_products.append({
        'A': [(0,0,1), (0,1,1)],
        'B': [(1,1,1)],
        'C': [(0,0,-1), (0,1,1)]
    })
    
    # Product 6: M6 = (c-d)*(f+h)
    # A: c(1,0) - d(1,1), B: f(0,1) + h(1,1)
    # C: C[1,1] += 1
    strassen_products.append({
        'A': [(1,0,1), (1,1,-1)],
        'B': [(0,1,1), (1,1,1)],
        'C': [(1,1,1)]
    })
    
    # Product 7: M7 = (b-c)*(e+g)
    # A: b(0,1) - c(1,0), B: e(0,0) + g(1,0)
    # C: C[0,0] += 1, C[1,0] += -1
    strassen_products.append({
        'A': [(0,1,1), (1,0,-1)],
        'B': [(0,0,1), (1,0,1)],
        'C': [(0,0,1), (1,0,-1)]
    })
    
    # Total rank = num_blocks^2 * len(strassen_products) = 9 * 7 = 63
    rank = num_blocks * num_blocks * len(strassen_products)
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 63)
    V = [[0] * rank for _ in range(m * p)]  # (36, 63)
    W = [[0] * rank for _ in range(p * n)]  # (36, 63)
    
    col = 0
    for u in range(num_blocks):  # block row of A and C
        for v in range(num_blocks):  # block col of A and block row of B
            for w in range(num_blocks):  # block col of B and C
                for prod in strassen_products:
                    # A block is at rows u*block_size.., cols v*block_size..
                    # For each (r_a, c_a, coeff_a) in prod['A']:
                    #   global row in A: u*block_size + r_a
                    #   global col in A: v*block_size + c_a
                    #   U index: (u*block_size + r_a) * m + (v*block_size + c_a)
                    for (r_a, c_a, coeff_a) in prod['A']:
                        a_row = u * block_size + r_a
                        a_col = v * block_size + c_a
                        U[a_row * m + a_col][col] += coeff_a
                    
                    # B block is at rows v*block_size.., cols w*block_size..
                    # For each (r_b, c_b, coeff_b) in prod['B']:
                    #   global row in B: v*block_size + r_b
                    #   global col in B: w*block_size + c_b
                    #   V index: (v*block_size + r_b) * p + (w*block_size + c_b)
                    for (r_b, c_b, coeff_b) in prod['B']:
                        b_row = v * block_size + r_b
                        b_col = w * block_size + c_b
                        V[b_row * p + b_col][col] += coeff_b
                    
                    # C block is at rows u*block_size.., cols w*block_size..
                    # For each (r_c, c_c, coeff_c) in prod['C']:
                    #   global row in C: u*block_size + r_c
                    #   global col in C: w*block_size + c_c
                    #   W index: c_c * n + r_c (per trace convention W[k*n+i, r])
                    for (r_c, c_c, coeff_c) in prod['C']:
                        c_row = u * block_size + r_c
                        c_col = w * block_size + c_c
                        W[c_col * n + c_row][col] += coeff_c
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

