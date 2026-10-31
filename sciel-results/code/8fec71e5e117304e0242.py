"""4x4 matrix multiplication tensor decomposition using recursive Strassen."""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 matrix multiplication using recursive Strassen.
    
    Rank 49 is the known minimal rank for 4x4 matrix multiplication over R.
    This implementation constructs the factor matrices explicitly.
    """
    n, m, p = DIMENSIONS  # all 4
    
    # Strassen decomposition for 2x2 matrix multiplication.
    # m1 = (X00 + X11) * (Y00 + Y11)
    # m2 = (X10 + X11) * Y00
    # m3 = X00 * (Y01 - Y11)
    # m4 = X11 * (Y10 - Y00)
    # m5 = (X00 + X01) * Y11
    # m6 = (X10 - X00) * (Y00 + Y01)
    # m7 = (X01 - X11) * (Y10 + Y11)
    
    # Factors for the inner 2x2 Strassen step
    # U: (coeff, row, col) for X
    strassen_U = [
        [(1, 0, 0), (1, 1, 1)],       # m1
        [(1, 1, 0), (1, 1, 1)],       # m2
        [(1, 0, 0)],                   # m3
        [(1, 1, 1)],                   # m4
        [(1, 0, 0), (1, 0, 1)],       # m5
        [(1, 1, 0), (-1, 0, 0)],      # m6
        [(1, 0, 1), (-1, 1, 1)],      # m7
    ]
    
    # V: (coeff, row, col) for Y
    strassen_V = [
        [(1, 0, 0), (1, 1, 1)],       # m1
        [(1, 0, 0)],                   # m2
        [(1, 0, 1), (-1, 1, 1)],      # m3
        [(1, 1, 0), (-1, 0, 0)],      # m4
        [(1, 1, 1)],                   # m5
        [(1, 0, 0), (1, 0, 1)],       # m6
        [(1, 1, 0), (1, 1, 1)],       # m7
    ]
    
    # W: (coeff, row, col) for Z
    strassen_W = [
        [(1, 0, 0), (1, 1, 1)],       # m1
        [(1, 1, 0), (-1, 1, 1)],      # m2
        [(1, 0, 1), (1, 1, 1)],       # m3
        [(1, 0, 0), (1, 1, 0)],       # m4
        [(-1, 0, 0), (1, 0, 1)],      # m5
        [(1, 1, 1)],                   # m6
        [(1, 0, 0)],                   # m7
    ]
    
    # Outer Strassen factors (same structure as inner, applied to blocks)
    outer_strassen_U = strassen_U
    outer_strassen_V = strassen_V
    outer_strassen_W = strassen_W
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    col = 0
    for outer_s in range(7):
        for inner_s in range(7):
            # Process U
            for (oc, obr, obc) in outer_strassen_U[outer_s]:
                for (ic, ibr, ibc) in strassen_U[inner_s]:
                    gr = 2 * obr + ibr
                    gc = 2 * obc + ibc
                    U[gr * m + gc][col] += oc * ic
            
            # Process V
            for (oc, obr, obc) in outer_strassen_V[outer_s]:
                for (ic, ibr, ibc) in strassen_V[inner_s]:
                    gr = 2 * obr + ibr
                    gc = 2 * obc + ibc
                    V[gr * p + gc][col] += oc * ic
            
            # Process W
            # W index is k*n + i, where k is col of result block, i is row of result block
            # In our block structure:
            # Result block (obr, obc) corresponds to global rows 2*obr..2*obr+1 and cols 2*obc..2*obc+1
            # W[k*n+i] where k is global col, i is global row.
            for (oc, obr, obc) in outer_strassen_W[outer_s]:
                for (ic, ibr, ibc) in strassen_W[inner_s]:
                    gr = 2 * obr + ibr
                    gc = 2 * obc + ibc
                    W[gc * n + gr][col] += oc * ic
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
