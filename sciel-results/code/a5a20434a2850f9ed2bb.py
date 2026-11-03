"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 78 decomposition
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def get_u_idx(i, j):
        return i * m + j

    def get_v_idx(j, k):
        return j * p + k

    def get_w_idx(k, i):
        return k * n + i

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, coeff in u_coeffs:
            if coeff != 0:
                U[idx][col] += coeff
        for idx, coeff in v_coeffs:
            if coeff != 0:
                V[idx][col] += coeff
        for idx, coeff in w_coeffs:
            if coeff != 0:
                W[idx][col] += coeff
        col += 1

    def add_strassen_222(i0, j0, k0):
        # A rows: i0, i0+1; cols: j0, j0+1
        # B rows: j0, j0+1; cols: k0, k0+1
        # C rows: i0, i0+1; cols: k0, k0+1
        
        A00 = get_u_idx(i0, j0)
        A01 = get_u_idx(i0, j0 + 1)
        A10 = get_u_idx(i0 + 1, j0)
        A11 = get_u_idx(i0 + 1, j0 + 1)
        
        B00 = get_v_idx(j0, k0)
        B01 = get_v_idx(j0, k0 + 1)
        B10 = get_v_idx(j0 + 1, k0)
        B11 = get_v_idx(j0 + 1, k0 + 1)
        
        C00 = get_w_idx(k0, i0)
        C01 = get_w_idx(k0, i0 + 1)
        C10 = get_w_idx(k0 + 1, i0)
        C11 = get_w_idx(k0 + 1, i0 + 1)
        
        # M1 = (A00 + A11)(B00 + B11)
        add_term([(A00, 1), (A11, 1)], [(B00, 1), (B11, 1)], [(C00, 1), (C11, 1)])
        
        # M2 = (A10 + A11)B00
        add_term([(A10, 1), (A11, 1)], [(B00, 1)], [(C10, 1), (C11, -1)])
        
        # M3 = A00(B01 - B11)
        add_term([(A00, 1)], [(B01, 1), (B11, -1)], [(C01, 1), (C11, 1)])
        
        # M4 = A11(B10 - B00)
        add_term([(A11, 1)], [(B10, 1), (B00, -1)], [(C00, 1), (C10, 1)])
        
        # M5 = (A00 + A01)B11
        add_term([(A00, 1), (A01, 1)], [(B11, 1)], [(C00, -1), (C01, 1)])
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term([(A10, 1), (A00, -1)], [(B00, 1), (B01, 1)], [(C11, 1)])
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term([(A01, 1), (A11, -1)], [(B10, 1), (B11, 1)], [(C00, 1)])

    def add_schoolbook_221(i0, j0, k0):
        # A 2x2, B 2x1, C 2x1
        A00 = get_u_idx(i0, j0)
        A01 = get_u_idx(i0, j0 + 1)
        A10 = get_u_idx(i0 + 1, j0)
        A11 = get_u_idx(i0 + 1, j0 + 1)
        
        B0 = get_v_idx(j0, k0)
        B1 = get_v_idx(j0 + 1, k0)
        
        C0 = get_w_idx(k0, i0)
        C1 = get_w_idx(k0, i0 + 1)
        
        # C0 = A00*B0 + A01*B1
        add_term([(A00, 1)], [(B0, 1)], [(C0, 1)])
        add_term([(A01, 1)], [(B1, 1)], [(C0, 1)])
        
        # C1 = A10*B0 + A11*B1
        add_term([(A10, 1)], [(B0, 1)], [(C1, 1)])
        add_term([(A11, 1)], [(B1, 1)], [(C1, 1)])

    def add_schoolbook_122(i0, j0, k0):
        # A 1x2, B 2x2, C 1x2
        A0 = get_u_idx(i0, j0)
        A1 = get_u_idx(i0, j0 + 1)
        
        B00 = get_v_idx(j0, k0)
        B01 = get_v_idx(j0, k0 + 1)
        B10 = get_v_idx(j0 + 1, k0)
        B11 = get_v_idx(j0 + 1, k0 + 1)
        
        C0 = get_w_idx(k0, i0)
        C1 = get_w_idx(k0 + 1, i0)
        
        # C0 = A0*B00 + A1*B10
        add_term([(A0, 1)], [(B00, 1)], [(C0, 1)])
        add_term([(A1, 1)], [(B10, 1)], [(C0, 1)])
        
        # C1 = A0*B01 + A1*B11
        add_term([(A0, 1)], [(B01, 1)], [(C1, 1)])
        add_term([(A1, 1)], [(B11, 1)], [(C1, 1)])

    def add_schoolbook_121(i0, j0, k0):
        # A 1x2, B 2x1, C 1x1
        A0 = get_u_idx(i0, j0)
        A1 = get_u_idx(i0, j0 + 1)
        
        B0 = get_v_idx(j0, k0)
        B1 = get_v_idx(j0 + 1, k0)
        
        C0 = get_w_idx(k0, i0)
        
        # C0 = A0*B0 + A1*B1
        add_term([(A0, 1)], [(B0, 1)], [(C0, 1)])
        add_term([(A1, 1)], [(B1, 1)], [(C0, 1)])

    # Top Part (i=0,1)
    # j=0,1; k=0,1
    add_strassen_222(0, 0, 0)
    # j=0,1; k=2,3
    add_strassen_222(0, 0, 2)
    # j=2,3; k=0,1
    add_strassen_222(0, 2, 0)
    # j=2,3; k=2,3
    add_strassen_222(0, 2, 2)
    # j=0,1; k=4,5
    add_strassen_222(0, 0, 4)
    # j=2,3; k=4,5
    add_strassen_222(0, 2, 4)
    # j=0,1; k=6
    add_schoolbook_221(0, 0, 6)
    # j=2,3; k=6
    add_schoolbook_221(0, 2, 6)
    
    # Bottom Part (i=2)
    # j=0,1; k=0,1
    add_schoolbook_122(2, 0, 0)
    # j=2,3; k=0,1
    add_schoolbook_122(2, 2, 0)
    # j=0,1; k=2,3
    add_schoolbook_122(2, 0, 2)
    # j=2,3; k=2,3
    add_schoolbook_122(2, 2, 2)
    # j=0,1; k=4,5
    add_schoolbook_122(2, 0, 4)
    # j=2,3; k=4,5
    add_schoolbook_122(2, 2, 4)
    # j=0,1; k=6
    add_schoolbook_121(2, 0, 6)
    # j=2,3; k=6
    add_schoolbook_121(2, 2, 6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
