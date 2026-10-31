# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 matrix multiplication using recursive Strassen.
    Rank = 49.
    """
    n, m, p = 4, 4, 4
    
    # Strassen factors for 2x2
    # U: (coeff, row, col) for A
    sU = [
        [(1, 0, 0), (1, 1, 1)],
        [(1, 1, 0), (1, 1, 1)],
        [(1, 0, 0)],
        [(1, 1, 1)],
        [(1, 0, 0), (1, 0, 1)],
        [(1, 1, 0), (-1, 0, 0)],
        [(1, 0, 1), (-1, 1, 1)]
    ]
    # V: (coeff, row, col) for B
    sV = [
        [(1, 0, 0), (1, 1, 1)],
        [(1, 0, 0)],
        [(1, 0, 1), (-1, 1, 1)],
        [(1, 1, 0), (-1, 0, 0)],
        [(1, 1, 1)],
        [(1, 0, 0), (1, 0, 1)],
        [(1, 1, 0), (1, 1, 1)]
    ]
    # W: (coeff, row, col) for C
    sW = [
        [(1, 0, 0), (1, 1, 1)],
        [(1, 1, 0), (-1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(1, 0, 0), (1, 1, 0)],
        [(-1, 0, 0), (1, 0, 1)],
        [(1, 1, 1)],
        [(1, 0, 0)]
    ]
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    col = 0
    for i in range(7):
        for j in range(7):
            # Outer i, Inner j
            # U
            for (ci, ri, ci_col) in sU[i]:
                for (cj, rj, cj_col) in sU[j]:
                    gr = 2*ri + rj
                    gc = 2*ci_col + cj_col
                    U[gr*m + gc][col] += ci * cj
            # V
            for (ci, ri, ci_col) in sV[i]:
                for (cj, rj, cj_col) in sV[j]:
                    gr = 2*ri + rj
                    gc = 2*ci_col + cj_col
                    V[gr*p + gc][col] += ci * cj
            # W
            for (ci, ri, ci_col) in sW[i]:
                for (cj, rj, cj_col) in sW[j]:
                    gr = 2*ri + rj
                    gc = 2*ci_col + cj_col
                    W[gc*n + gr][col] += ci * cj
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
