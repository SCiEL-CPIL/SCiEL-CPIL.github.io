"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (22, rank)
    # W: (p*n, rank) = (22, rank)
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] = val
        for idx, val in v_coeffs.items():
            V[idx][col] = val
        for idx, val in w_coeffs.items():
            W[idx][col] = val
        col += 1
        
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: j*p + k. j in {0,1}, k in {0..10}
    # W indices: k*n + i. k in {0..10}, i in {0,1}
    
    # 5 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9)
    for b in range(5):
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # B indices
        b00 = 0 * p + c0
        b01 = 0 * p + c1
        b10 = 1 * p + c0
        b11 = 1 * p + c1
        
        # W indices
        w00 = c0 * n + 0
        w01 = c1 * n + 0
        w10 = c0 * n + 1
        w11 = c1 * n + 1
        
        # Strassen M1: (A00 + A11) * (B00 + B11)
        # C00 += M1, C11 += M1
        add_term(
            {0: 1, 3: 1},
            {b00: 1, b11: 1},
            {w00: 1, w11: 1}
        )
        
        # Strassen M2: (A10 + A11) * B00
        # C10 += M2, C11 -= M2
        add_term(
            {2: 1, 3: 1},
            {b00: 1},
            {w10: 1, w11: -1}
        )
        
        # Strassen M3: A00 * (B10 - B11)
        # C01 += M3, C11 += M3
        add_term(
            {0: 1},
            {b10: 1, b11: -1},
            {w01: 1, w11: 1}
        )
        
        # Strassen M4: A11 * (B00 - B01)
        # C00 += M4, C10 += M4
        add_term(
            {3: 1},
            {b00: 1, b01: -1},
            {w00: 1, w10: 1}
        )
        
        # Strassen M5: (A00 + A01) * B11
        # C00 -= M5, C01 += M5
        add_term(
            {0: 1, 1: 1},
            {b11: 1},
            {w00: -1, w01: 1}
        )
        
        # Strassen M6: (A10 - A00) * (B00 + B01)
        # C11 += M6
        add_term(
            {2: 1, 0: -1},
            {b00: 1, b01: 1},
            {w11: 1}
        )
        
        # Strassen M7: (A01 - A11) * (B10 + B11)
        # C00 += M7
        add_term(
            {1: 1, 3: -1},
            {b10: 1, b11: 1},
            {w00: 1}
        )
        
    # Last column 10: Schoolbook
    c = 10
    # B indices
    b0 = 0 * p + c
    b1 = 1 * p + c
    # W indices
    w0 = c * n + 0
    w1 = c * n + 1
    
    # C00 = A00*B0 + A01*B1
    add_term({0: 1}, {b0: 1}, {w0: 1})
    add_term({1: 1}, {b1: 1}, {w0: 1})
    
    # C10 = A10*B0 + A11*B1
    add_term({2: 1}, {b0: 1}, {w1: 1})
    add_term({3: 1}, {b1: 1}, {w1: 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
