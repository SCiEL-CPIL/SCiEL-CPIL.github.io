DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(i0, j0, k0, col_offset):
        nonlocal col
        # Strassen coefficients
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10 += M2
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # 2x5x6 block (i=0,1)
    # Split j=0..4 into {0,1}, {2,3}, {4}
    # For j in {0,1}, split k=0..5 into {0,1}, {2,3}, {4,5}
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_block(0, j0, k0, col)
            add_strassen_block(1, j0, k0, col) # Wait, i0 is 0 for i=0, 1 for i=1?
            # No, the block is 2x2x2. i0 covers both i=0 and i=1.
            # So we just call it once for i0=0?
            # The function add_strassen_block(i0, j0, k0) handles i0 and i0+1.
            # So for i in {0,1}, we set i0=0.
            # But we have two j-groups: {0,1} and {2,3}.
            # So we call add_strassen_block(0, 0, k0) and add_strassen_block(0, 2, k0).
            pass

    # Correct loop structure:
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_block(0, j0, k0, col)
            
    # j=4 block (2x1x6)
    # i=0,1; j=4; k=0..5
    # Rank 2
    # r=0: i=0
    U[0*m + 4][col] += 1
    V[4*p + 0][col] += 1 # V index 4*6+0? No, V is 5x6. j=4, k=0..5.
    # We need V_{4,k,r} = 1 for all k.
    for k in range(6):
        V[4*p + k][col] += 1
    for k in range(6):
        W[k*n + 0][col] += 1
    col += 1
    
    # r=1: i=1
    U[1*m + 4][col] += 1
    for k in range(6):
        V[4*p + k][col] += 1
    for k in range(6):
        W[k*n + 1][col] += 1
    col += 1

    # 1x5x6 block (i=2)
    # Rank 5
    for r in range(5):
        j = r
        U[2*m + j][col] += 1
        for k in range(6):
            V[j*p + k][col] += 1
        for k in range(6):
            W[k*n + 2][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
