import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate
    # 6 Strassen blocks (2x2x2) -> 6 * 7 = 42
    # 3 Schoolbook blocks (2x1x2) -> 3 * 4 = 12
    # Total 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add Strassen block
    # i0, j0, k0 are the starting indices
    def add_strassen(i0, j0, k0):
        nonlocal col_idx
        # 7 terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for U (A part), V (B part), W (C part)
        # U indices: (i0+di, j0+dj)
        # V indices: (j0+dj, k0+dk)
        # W indices: (k0+dk, i0+di) -> global index (k0+dk)*n + (i0+di)
        
        terms = [
            # (u_coefs, v_coefs, w_coefs)
            # u_coefs: list of (di, dj, val)
            # v_coefs: list of (dj, dk, val)
            # w_coefs: list of (dk, di, val)
            
            # M1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1)]), # W: C10 -> k=0, i=1
            # M3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1)]), # W: C01 -> k=1, i=0
            # M4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]), # W: C00, C10
            # M5
            ([(0,0,1), (1,0,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]), # W: C00(-1), C01(1)
            # M6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]), # W: C11
            # M7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]), # W: C00
        ]
        
        for u_coefs, v_coefs, w_coefs in terms:
            c = col_idx
            for di, dj, val in u_coefs:
                U[(i0+di)*m + (j0+dj)][c] = val
            for dj, dk, val in v_coefs:
                V[(j0+dj)*p + (k0+dk)][c] = val
            for dk, di, val in w_coefs:
                W[(k0+dk)*n + (i0+di)][c] = val
            col_idx += 1

    # Helper to add schoolbook block
    def add_schoolbook(i0, j0, k0, ni, nj, nk):
        nonlocal col_idx
        for di in range(ni):
            for dj in range(nj):
                for dk in range(nk):
                    c = col_idx
                    U[(i0+di)*m + (j0+dj)][c] = 1
                    V[(j0+dj)*p + (k0+dk)][c] = 1
                    W[(k0+dk)*n + (i0+di)][c] = 1
                    col_idx += 1

    # Apply Strassen blocks
    # j blocks: 0-1, 2-3. k blocks: 0-1, 2-3, 4-5. i is always 0-1.
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 2, 4)
    
    # Apply Schoolbook blocks for j=4
    # j=4, k=0-1
    add_schoolbook(0, 4, 0, 2, 1, 2)
    # j=4, k=2-3
    add_schoolbook(0, 4, 2, 2, 1, 2)
    # j=4, k=4-5
    add_schoolbook(0, 4, 4, 2, 1, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
