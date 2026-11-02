"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 7 blocks of width 2, 1 block of width 1
    # Rank = 7 * 7 + 4 = 53
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 multiplication
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Contributions to C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U indices: i*m + j. i in 0..1, j in 0..1.
    # V indices: j*p + k. j in 0..1, k in 0..14.
    # W indices: k*n + i. k in 0..14, i in 0..1.
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs, col):
        # u_coeffs: list of (idx, val) for U
        # v_coeffs: list of (idx, val) for V
        # w_coeffs: list of (idx, val) for W
        for idx, val in u_coeffs:
            U[idx][col] += val
        for idx, val in v_coeffs:
            V[idx][col] += val
        for idx, val in w_coeffs:
            W[idx][col] += val

    col = 0
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        # U: A00 (idx 0), A11 (idx 3)
        # V: B0k0 (idx 0*p + k0), B1k1 (idx 1*p + k1)
        # W: C0k0 (idx k0*n + 0), C1k1 (idx k1*n + 1)
        # C00 coeff 1, C11 coeff 1.
        # W contributions:
        # C00 is at i=0, k=k0. W index k0*n + 0.
        # C11 is at i=1, k=k1. W index k1*n + 1.
        
        add_term(
            [(0, 1), (3, 1)],  # U: A00 + A11
            [(k0, 1), (p + k1, 1)], # V: B0k0 + B1k1
            [(k0 * n + 0, 1), (k1 * n + 1, 1)], # W: C0k0 + C1k1
            col
        )
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        # U: A10 (idx 2), A11 (idx 3)
        # V: B0k0 (idx k0)
        # W: C10 coeff 1, C11 coeff -1.
        # C10 is i=1, k=k0. W index k0*n + 1.
        # C11 is i=1, k=k1. W index k1*n + 1.
        
        add_term(
            [(2, 1), (3, 1)],  # U: A10 + A11
            [(k0, 1)], # V: B0k0
            [(k0 * n + 1, 1), (k1 * n + 1, -1)], # W: C10 - C11
            col
        )
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        # U: A00 (idx 0)
        # V: B0k1 (idx p+k1), B1k1 (idx p+k1) -> Wait, B1k1 is idx 1*p + k1.
        # V: B0k1 (idx k1), B1k1 (idx p+k1).
        # W: C01 coeff 1, C11 coeff 1.
        # C01 is i=0, k=k1. W index k1*n + 0.
        # C11 is i=1, k=k1. W index k1*n + 1.
        
        add_term(
            [(0, 1)], # U: A00
            [(k1, 1), (p + k1, -1)], # V: B0k1 - B1k1
            [(k1 * n + 0, 1), (k1 * n + 1, 1)], # W: C01 + C11
            col
        )
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        # U: A11 (idx 3)
        # V: B1k0 (idx p+k0), B0k0 (idx k0).
        # W: C00 coeff 1, C10 coeff 1.
        # C00 is i=0, k=k0. W index k0*n + 0.
        # C10 is i=1, k=k0. W index k0*n + 1.
        
        add_term(
            [(3, 1)], # U: A11
            [(p + k0, 1), (k0, -1)], # V: B1k0 - B0k0
            [(k0 * n + 0, 1), (k0 * n + 1, 1)], # W: C00 + C10
            col
        )
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        # U: A00 (idx 0), A01 (idx 1).
        # V: B1k1 (idx p+k1).
        # W: C00 coeff -1, C01 coeff 1.
        # C00 is i=0, k=k0. W index k0*n + 0.
        # C01 is i=0, k=k1. W index k1*n + 0.
        
        add_term(
            [(0, 1), (1, 1)], # U: A00 + A01
            [(p + k1, 1)], # V: B1k1
            [(k0 * n + 0, -1), (k1 * n + 0, 1)], # W: -C00 + C01
            col
        )
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        # U: A10 (idx 2), A00 (idx 0).
        # V: B0k0 (idx k0), B0k1 (idx k1).
        # W: C11 coeff 1.
        # C11 is i=1, k=k1. W index k1*n + 1.
        
        add_term(
            [(2, 1), (0, -1)], # U: A10 - A00
            [(k0, 1), (k1, 1)], # V: B0k0 + B0k1
            [(k1 * n + 1, 1)], # W: C11
            col
        )
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        # U: A01 (idx 1), A11 (idx 3).
        # V: B1k0 (idx p+k0), B1k1 (idx p+k1).
        # W: C00 coeff 1.
        # C00 is i=0, k=k0. W index k0*n + 0.
        
        add_term(
            [(1, 1), (3, -1)], # U: A01 - A11
            [(p + k0, 1), (p + k1, 1)], # V: B1k0 + B1k1
            [(k0 * n + 0, 1)], # W: C00
            col
        )
        col += 1
        
    # Last block: column k=14, schoolbook
    k_last = 14
    
    # T1: A00 * B0k -> C0k
    add_term(
        [(0, 1)],
        [(k_last, 1)],
        [(k_last * n + 0, 1)],
        col
    )
    col += 1
    
    # T2: A01 * B1k -> C0k
    add_term(
        [(1, 1)],
        [(p + k_last, 1)],
        [(k_last * n + 0, 1)],
        col
    )
    col += 1
    
    # T3: A10 * B0k -> C1k
    add_term(
        [(2, 1)],
        [(k_last, 1)],
        [(k_last * n + 1, 1)],
        col
    )
    col += 1
    
    # T4: A11 * B1k -> C1k
    add_term(
        [(3, 1)],
        [(p + k_last, 1)],
        [(k_last * n + 1, 1)],
        col
    )
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
