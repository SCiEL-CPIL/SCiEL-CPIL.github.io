"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    col = 0
    
    # Block 1: k in {0, 1}
    # Sub-block 1a: j in {0, 1}, k in {0, 1} (Strassen)
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=0, B01=1, B10=4, B11=5
    # C indices: C00=0, C01=2, C10=1, C11=3
    
    # M1
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], col); col += 1
    # M2
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], col); col += 1
    # M3
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], col); col += 1
    # M4
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], col); col += 1
    # M5
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], col); col += 1
    # M6
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], col); col += 1
    # M7
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1), (3, -1)], col); col += 1
    
    # Sub-block 1b: j=2, k in {0, 1} (Schoolbook)
    # A indices: A02=2, A12=5
    # B indices: B20=8, B21=9
    # C indices: C00=0, C01=2, C10=1, C11=3
    
    add_term([(2, 1)], [(8, 1)], [(0, 1)], col); col += 1
    add_term([(2, 1)], [(9, 1)], [(2, 1)], col); col += 1
    add_term([(5, 1)], [(8, 1)], [(1, 1)], col); col += 1
    add_term([(5, 1)], [(9, 1)], [(3, 1)], col); col += 1
    
    # Block 2: k in {2, 3}
    # Sub-block 2a: j in {0, 1}, k in {2, 3} (Strassen)
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B02=2, B03=3, B12=6, B13=7
    # C indices: C02=4, C03=6, C12=5, C13=7
    
    # M1
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], col); col += 1
    # M2
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], col); col += 1
    # M3
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], col); col += 1
    # M4
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], col); col += 1
    # M5
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], col); col += 1
    # M6
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], col); col += 1
    # M7
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1), (7, -1)], col); col += 1
    
    # Sub-block 2b: j=2, k in {2, 3} (Schoolbook)
    # A indices: A02=2, A12=5
    # B indices: B22=10, B23=11
    # C indices: C02=4, C03=6, C12=5, C13=7
    
    add_term([(2, 1)], [(10, 1)], [(4, 1)], col); col += 1
    add_term([(2, 1)], [(11, 1)], [(6, 1)], col); col += 1
    add_term([(5, 1)], [(10, 1)], [(5, 1)], col); col += 1
    add_term([(5, 1)], [(11, 1)], [(7, 1)], col); col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
