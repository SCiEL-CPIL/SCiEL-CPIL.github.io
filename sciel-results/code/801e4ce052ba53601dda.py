"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_coeffs: list of (idx, coeff) for U
    # v_coeffs: list of (idx, coeff) for V
    # w_coeffs: list of (idx, coeff) for W
    def add_term(col, u_coeffs, v_coeffs, w_coeffs):
        for idx, coeff in u_coeffs:
            U[idx][col] += coeff
        for idx, coeff in v_coeffs:
            V[idx][col] += coeff
        for idx, coeff in w_coeffs:
            W[idx][col] += coeff

    # Block 1: B columns 0, 1 (e, f, g, h)
    # V indices: e=0, f=1, g=4, h=5
    # W indices: C00=0, C10=1, C01=2, C11=3
    
    # M1 = (a+d)(e+h)
    add_term(0, [(0, 1), (3, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)])
    
    # M2 = (c+d)e
    add_term(1, [(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)])
    
    # M3 = a(f-h)
    add_term(2, [(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)])
    
    # M4 = d(g-e)
    add_term(3, [(3, 1)], [(4, 1), (0, -1)], [(0, -1), (1, 1)])
    
    # M5 = (a+b)h
    add_term(4, [(0, 1), (1, 1)], [(5, 1)], [(0, 1), (2, 1)])
    
    # M6 = (c-d)(f+h)
    add_term(5, [(2, 1), (3, -1)], [(1, 1), (5, 1)], [(3, 1)])
    
    # M7 = (b-c)(g+e)
    add_term(6, [(1, 1), (2, -1)], [(4, 1), (0, 1)], [(0, -1)])
    
    # Block 2: B columns 2, 3 (i2, j2, k2, l2)
    # V indices: i2=2, j2=3, k2=6, l2=7
    # W indices: C02=4, C12=5, C03=6, C13=7
    
    # N1 = (a+d)(i2+l2)
    add_term(7, [(0, 1), (3, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)])
    
    # N2 = (c+d)i2
    add_term(8, [(2, 1), (3, 1)], [(2, 1)], [(5, 1), (7, -1)])
    
    # N3 = a(j2-l2)
    add_term(9, [(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)])
    
    # N4 = d(k2-i2)
    add_term(10, [(3, 1)], [(6, 1), (2, -1)], [(4, -1), (5, 1)])
    
    # N5 = (a+b)l2
    add_term(11, [(0, 1), (1, 1)], [(7, 1)], [(4, 1), (6, 1)])
    
    # N6 = (c-d)(j2+l2)
    add_term(12, [(2, 1), (3, -1)], [(3, 1), (7, 1)], [(7, 1)])
    
    # N7 = (b-c)(k2+i2)
    add_term(13, [(1, 1), (2, -1)], [(6, 1), (2, 1)], [(4, -1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
