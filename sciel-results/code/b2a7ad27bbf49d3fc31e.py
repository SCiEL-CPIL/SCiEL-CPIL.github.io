import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 6 blocks of size 2x2 (rank 7 each) and 1 block of size 2x1 (rank 4)
    # Total rank = 6 * 7 + 4 = 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_indices_vals, v_indices_vals, w_indices_vals):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_indices_vals:
            U[idx][col] += val
        for idx, val in v_indices_vals:
            V[idx][col] += val
        for idx, val in w_indices_vals:
            W[idx][col] += val
        col += 1

    # Strassen decomposition for 2x2 blocks
    # A is 2x2, B is 2x2 (columns 2k, 2k+1)
    # C is 2x2 (columns 2k, 2k+1)
    
    # Indices for A: i*m + j
    # Indices for B: j*p + k
    # Indices for W: k*n + i
    
    for block in range(6):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # M1 = (a00 + a11)(b00 + b11)
        # Here b00 is B[0, k0], b11 is B[1, k1]
        add_term(
            [(0*m+0, 1), (1*m+1, 1)],
            [(0*p+k0, 1), (1*p+k1, 1)],
            [(k0*n+0, 1), (k1*n+0, -1), (k0*n+1, -1), (k1*n+1, 1)] # c00=+m1, c01=-m1, c10=-m1, c11=+m1? 
            # Let's re-verify W coefficients for M1
            # c00 = m1 + m4 - m5 + m7
            # c01 = -m1 + m3 + m5 + m6
            # c10 = -m2 + m1 + m4 + m7
            # c11 = m2 - m3 - m4 + m6
            # So for M1:
            # c00: +1
            # c01: -1
            # c10: +1
            # c11: 0
            # Wait, standard Strassen:
            # C00 = M1 + M4 - M5 + M7
            # C01 = -M1 + M3 + M5 + M6
            # C10 = -M2 + M1 + M4 + M7
            # C11 = M2 - M3 - M4 + M6
            # So M1 coeffs: C00=1, C01=-1, C10=1, C11=0.
            # My W indices: k*n+i
            # C00 -> k0*n+0
            # C01 -> k1*n+0
            # C10 -> k0*n+1
            # C11 -> k1*n+1
        )
        # Correction: I need to be careful with the W coefficients for each M.
        
        # Let's define the 7 terms explicitly with their W coefficients.
        
        # M1
        add_term(
            [(0, 1), (3, 1)], # a00, a11
            [(k0, 1), (p+k1, 1)], # b00, b11 (indices: 0*p+k0, 1*p+k1)
            [(k0*n+0, 1), (k1*n+0, -1), (k0*n+1, 1), (k1*n+1, 0)]
        )
        
        # M2 = (a10 + a11)b00
        add_term(
            [(2, 1), (3, 1)], # a10, a11
            [(k0, 1)], # b00
            [(k0*n+0, 0), (k1*n+0, 0), (k0*n+1, -1), (k1*n+1, 1)]
            # C10 coeff for M2 is -1. C11 coeff is 1.
            # C10 -> k0*n+1. C11 -> k1*n+1.
        )
        
        # M3 = a00(b10 - b11)
        add_term(
            [(0, 1)], # a00
            [(p+k0, 1), (p+k1, -1)], # b10, b11
            [(k0*n+0, 0), (k1*n+0, 1), (k0*n+1, 0), (k1*n+1, 0)]
            # C01 coeff for M3 is 1. C01 -> k1*n+0.
        )
        
        # M4 = a11(b01 - b00)
        add_term(
            [(3, 1)], # a11
            [(p+k1, 1), (k0, -1)], # b01, b00
            [(k0*n+0, 1), (k1*n+0, 0), (k0*n+1, 1), (k1*n+1, 0)]
            # C00 coeff for M4 is 1. C10 coeff for M4 is 1.
            # C00 -> k0*n+0. C10 -> k0*n+1.
        )
        
        # M5 = (a00 + a01)b11
        add_term(
            [(0, 1), (1, 1)], # a00, a01
            [(p+k1, 1)], # b11
            [(k0*n+0, -1), (k1*n+0, 1), (k0*n+1, 0), (k1*n+1, 0)]
            # C00 coeff for M5 is -1. C01 coeff for M5 is 1.
            # C00 -> k0*n+0. C01 -> k1*n+0.
        )
        
        # M6 = (a10 - a00)(b00 + b01)
        add_term(
            [(2, 1), (0, -1)], # a10, a00
            [(k0, 1), (p+k1, 1)], # b00, b01
            [(k0*n+0, 0), (k1*n+0, 1), (k0*n+1, 0), (k1*n+1, 1)]
            # C01 coeff for M6 is 1. C11 coeff for M6 is 1.
            # C01 -> k1*n+0. C11 -> k1*n+1.
        )
        
        # M7 = (a01 - a11)(b10 + b11)
        add_term(
            [(1, 1), (3, -1)], # a01, a11
            [(p+k0, 1), (p+k1, 1)], # b10, b11
            [(k0*n+0, 1), (k1*n+0, 0), (k0*n+1, 1), (k1*n+1, 0)]
            # C00 coeff for M7 is 1. C10 coeff for M7 is 1.
            # C00 -> k0*n+0. C10 -> k0*n+1.
        )
        
    # Last block: 2x1 (column 12)
    k = 12
    # Schoolbook:
    # C0 = a00*b0 + a01*b1
    # C1 = a10*b0 + a11*b1
    
    # T1 = a00 * b0
    add_term(
        [(0, 1)],
        [(k, 1)], # 0*p + k
        [(k*n+0, 1), (k*n+1, 0)]
    )
    
    # T2 = a01 * b1
    add_term(
        [(1, 1)],
        [(p+k, 1)], # 1*p + k
        [(k*n+0, 1), (k*n+1, 0)]
    )
    
    # T3 = a10 * b0
    add_term(
        [(2, 1)],
        [(k, 1)],
        [(k*n+0, 0), (k*n+1, 1)]
    )
    
    # T4 = a11 * b1
    add_term(
        [(3, 1)],
        [(p+k, 1)],
        [(k*n+0, 0), (k*n+1, 1)]
    )
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
