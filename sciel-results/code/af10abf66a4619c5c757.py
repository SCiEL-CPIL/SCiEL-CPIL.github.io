import time

def construct():
    n, m, p = 4, 4, 6
    t_start = time.time()
    
    # Build list of (coeff, i, j, k) terms
    terms = []
    
    # Helper: add a block product term
    # A_block (ra, ca) times B_block (rb, cb), coefficient c
    # Valid only if ca == rb
    # Contributes to C_block (ra, cb)
    def add_block_product(c, ra, ca, rb, cb):
        if ca != rb:
            return
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    i = 2*ra + ii
                    j = 2*ca + jj
                    k = 2*cb + kk
                    terms.append((c, i, j, k))
    
    # Strassen multipliers for 4x4x4
    # M1 = (A00+A11)*(B00+B11)
    add_block_product(1, 0, 0, 0, 0)  # A00*B00
    add_block_product(1, 1, 1, 1, 1)  # A11*B11
    # M2 = (A10+A11)*B00
    add_block_product(1, 1, 0, 0, 0)  # A10*B00
    # M3 = A00*(B01-B11)
    add_block_product(1, 0, 0, 0, 1)  # A00*B01
    # M4 = A11*(B10-B00)
    add_block_product(1, 1, 1, 1, 0)  # A11*B10
    # M5 = (A00+A01)*B11
    add_block_product(1, 0, 1, 1, 1)  # A01*B11
    # M6 = (A10-A00)*(B00+B01)
    add_block_product(1, 1, 0, 0, 0)  # A10*B00
    add_block_product(1, 1, 0, 0, 1)  # A10*B01
    add_block_product(-1, 0, 0, 0, 0)  # -A00*B00
    add_block_product(-1, 0, 0, 0, 1)  # -A00*B01
    # M7 = (A01-A11)*(B10+B11)
    add_block_product(1, 0, 1, 1, 0)  # A01*B10
    add_block_product(1, 0, 1, 1, 1)  # A01*B11
    add_block_product(-1, 1, 1, 1, 0)  # -A11*B10
    add_block_product(-1, 1, 1, 1, 1)  # -A11*B11
    
    # Now apply C-block formulas
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # But I've already added terms with their coefficients. The issue is that
    # the same (i,j,k) might appear multiple times with different coefficients
    # from different multipliers. I need to sum them up.
    
    # Actually, the terms list has individual scalar multiplications. Each one
    # is a rank-1 term in the tensor decomposition. But if the same (i,j,k)
    # appears multiple times, I should combine them into one term with summed
    # coefficient.
    
    # Let me collect all terms and combine.
    from collections import defaultdict
    term_map = defaultdict(float)
    for c, i, j, k in terms:
        if k < 4:  # Only 4x4x4 part
            term_map[(i, j, k)] += c
    
    # Now for the 4x4x2 part (k=4,5), use schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                term_map[(i, j, k)] += 1
    
    # Create U, V, W
    R = len(term_map)
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    
    col = 0
    for (i, j, k), c in term_map.items():
        U[i*m+j][col] = c
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
