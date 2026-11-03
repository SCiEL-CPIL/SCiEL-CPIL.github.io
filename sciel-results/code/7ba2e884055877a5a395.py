"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Build a list of (u_indices, v_indices, w_indices, coeffs) tuples
    # Each represents one tensor term: sum over t of U[i,t]*V[j,t]*W[k,t]
    # where U indices are for A, V for B, W for C.
    # U index: i*m + j
    # V index: j*p + k  
    # W index: k*n + i
    
    terms = []
    
    # Partition into 2x2 blocks for A, 2x4 for B, 2x4 for C
    # A: 2x2 blocks at (2*ai, 2*aj) for ai,aj in {0,1}
    # B: 2x4 blocks at (2*bi, 4*k) for bi in {0,1}, k in {0,1}
    # C: 2x4 blocks at (2*ai, 4*k) for ai in {0,1}, k in {0,1}
    
    for ai in range(2):
        for aj in range(2):
            for bi in range(2):
                for k in range(2):
                    # A subblock: [[a00,a01],[a10,a11]] at rows 2*ai..2*ai+1, cols 2*aj..2*aj+1
                    a00 = (2*ai) * m + (2*aj)
                    a01 = (2*ai) * m + (2*aj + 1)
                    a10 = (2*ai + 1) * m + (2*aj)
                    a11 = (2*ai + 1) * m + (2*aj + 1)
                    
                    # B subblock: 2 rows x 4 cols, rows 2*bi..2*bi+1, cols 4*k..4*k+3
                    b00 = (2*bi) * p + 4*k
                    b01 = (2*bi) * p + 4*k + 1
                    b02 = (2*bi) * p + 4*k + 2
                    b03 = (2*bi) * p + 4*k + 3
                    b10 = (2*bi + 1) * p + 4*k
                    b11 = (2*bi + 1) * p + 4*k + 1
                    b12 = (2*bi + 1) * p + 4*k + 2
                    b13 = (2*bi + 1) * p + 4*k + 3
                    
                    # C subblock: 2 rows x 4 cols, rows 2*ai..2*ai+1, cols 4*k..4*k+3
                    # W index: col * n + row
                    c00 = (4*k) * n + (2*ai)
                    c01 = (4*k + 1) * n + (2*ai)
                    c02 = (4*k + 2) * n + (2*ai)
                    c03 = (4*k + 3) * n + (2*ai)
                    c10 = (4*k) * n + (2*ai + 1)
                    c11 = (4*k + 1) * n + (2*ai + 1)
                    c12 = (4*k + 2) * n + (2*ai + 1)
                    c13 = (4*k + 3) * n + (2*ai + 1)
                    
                    # Apply Strassen to compute A_sub * B_sub
                    # For each pair of columns in B, apply 2x2*2x2 Strassen
                    # Left pair: B cols (b00,b01,b10,b11) -> C cols (c00,c01,c10,c11)
                    # Right pair: B cols (b02,b03,b12,b13) -> C cols (c02,c03,c12,c13)
                    
                    # Helper: add Strassen terms for 2x2*2x2
                    def add_strassen(a, b, c):
                        # a = [a00, a01, a10, a11], b = [b00, b01, b10, b11], c = [c00, c01, c10, c11]
                        # M1 = (a00+a11)*(b00+b11)
                        terms.append(([a[0], a[3]], [b[0], b[3]], [c[0], c[3], c[2]], [1, 1, -1]))
                        # M2 = (a10+a11)*b00
                        terms.append(([a[2], a[3]], [b[0]], [c[2], c[3]], [1, -1]))
                        # M3 = a00*(b11-b10)
                        terms.append(([a[0]], [b[3], b[2]], [c[1], c[3]], [1, 1]))
                        # M4 = a11*(b01-b00)
                        terms.append(([a[3]], [b[1], b[0]], [c[2], c[3]], [1, 1]))
                        # M5 = (a00+a01)*b11
                        terms.append(([a[0], a[1]], [b[3]], [c[1], c[0]], [1, -1]))
                        # M6 = (a10-a00)*(b00+b01)
                        terms.append(([a[2], a[0]], [b[0], b[1]], [c[2]], [1, 1]))
                        # M7 = (a01-a11)*(b10+b11)
                        terms.append(([a[1], a[3]], [b[2], b[3]], [c[0]], [1, -1]))
                    
                    # Left 2x2*2x2
                    add_strassen(
                        [a00, a01, a10, a11],
                        [b00, b01, b10, b11],
                        [c00, c01, c10, c11]
                    )
                    # Right 2x2*2x2
                    add_strassen(
                        [a00, a01, a10, a11],
                        [b02, b03, b12, b13],
                        [c02, c03, c12, c13]
                    )
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (u_idx, v_idx, w_idx, coeffs) in enumerate(terms):
        for ui, c in zip(u_idx, coeffs):
            U[ui][t] += c
        for vi, c in zip(v_idx, coeffs):
            V[vi][t] += c
        for wi, c in zip(w_idx, coeffs):
            W[wi][t] += c
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
