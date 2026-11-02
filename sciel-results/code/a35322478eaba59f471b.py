"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R = 54
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(j_start):
        nonlocal col
        # Coefficients for M1..M7
        # Format: (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: list of (ai, aj, val) where aj is relative to j_start
        # b_coeffs: list of (bj, bk_rel, val) where bj is relative to j_start, bk_rel is 0 or 1
        # c_coeffs: list of (ci, ck_rel, val) where ck_rel is 0 or 1
        
        terms = [
            # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2: (A10+A11)B00 -> C10+1, C11-1
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # M3: A00(B10-B11) -> C01+1, C11+1
            ([(0,0,1)], [(1,0,-1), (1,1,1)], [(0,1,1), (1,1,1)]),
            # M4: A11(B00-B01) -> C00+1, C10+1
            ([(1,1,1)], [(0,0,1), (0,1,-1)], [(0,0,1), (1,0,1)]),
            # M5: (A00+A01)B11 -> C00-1, C01+1
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # M6: (A10-A00)(B00+B01) -> C11+1
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # M7: (A01-A11)(B10+B11) -> C00+1
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        ]
        
        for s in range(3):
            k_curr = 2 * s
            for t in terms:
                a_coeffs, b_coeffs, c_coeffs = t
                for (ai, aj, val) in a_coeffs:
                    gi = ai
                    gj = j_start + aj
                    u_idx = gi * m + gj
                    U[u_idx][col] += val
                for (bj, bk_rel, val) in b_coeffs:
                    gj = j_start + bj
                    gk = k_curr + bk_rel
                    v_idx = gj * p + gk
                    V[v_idx][col] += val
                for (ci, ck_rel, val) in c_coeffs:
                    gi = ci
                    gk = k_curr + ck_rel
                    w_idx = gk * n + gi
                    W[w_idx][col] += val
                col += 1

    # Block 1: A1 B1 (j_start=0)
    add_strassen_block(0)
    
    # Block 2: A2 B2 (j_start=2)
    add_strassen_block(2)
    
    # Block 3: A3 B3 (j=4)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
