import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # s, u_coeffs (list of (idx, val)), v_coeffs (list of (idx, val)), w_coeffs (list of (idx, val))
    # Indices are relative to the block
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3 (relative to block start)
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (relative to block start)
    
    strassen_terms = [
        # s=0: M0 = (A00+A11)(B00+B11)
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': [(0, 1), (3, 1)]
        },
        # s=1: M1 = (A10+A11)B00
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': [(2, 1), (3, -1)]
        },
        # s=2: M2 = A00(B01-B11)
        {
            'u': [(0, 1)],
            'v': [(1, 1), (3, -1)],
            'w': [(1, 1), (3, 1)]
        },
        # s=3: M3 = A11(B10-B00)
        {
            'u': [(3, 1)],
            'v': [(2, 1), (0, -1)],
            'w': [(0, 1), (2, 1)]
        },
        # s=4: M4 = (A00+A01)B11
        {
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': [(0, -1), (1, 1)]
        },
        # s=5: M5 = (A10-A00)(B00+B01)
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 1), (1, 1)],
            'w': [(3, 1)]
        },
        # s=6: M6 = (A01-A11)(B10+B11)
        {
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': [(0, 1)]
        }
    ]
    
    for b in range(4):
        for s in range(7):
            c = b * 7 + s
            term = strassen_terms[s]
            
            # Update U
            for idx, val in term['u']:
                U[idx][c] += val
                
            # Update V
            for idx, val in term['v']:
                # idx is relative to block: 00, 01, 10, 11
                # Global V index: row * p + col
                # Relative idx: 0->00, 1->01, 2->10, 3->11
                r = idx // 2
                col_rel = idx % 2
                col_global = 2 * b + col_rel
                v_idx = r * p + col_global
                V[v_idx][c] += val
                
            # Update W
            for idx, val in term['w']:
                # idx is relative to block: 00, 01, 10, 11
                # Global W index: col * n + row
                # Relative idx: 0->00, 1->01, 2->10, 3->11
                # Note: W index convention is W[k*n+i, r] where k is col, i is row.
                # In Strassen C, 00 is row 0 col 0, 01 is row 0 col 1, 10 is row 1 col 0, 11 is row 1 col 1.
                # So relative idx 0 -> row 0, col 0
                # relative idx 1 -> row 0, col 1
                # relative idx 2 -> row 1, col 0
                # relative idx 3 -> row 1, col 1
                
                r = idx // 2
                col_rel = idx % 2
                col_global = 2 * b + col_rel
                w_idx = col_global * n + r
                W[w_idx][c] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
