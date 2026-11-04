"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # (8, 42)
    V = [[0] * rank for _ in range(m * p)]   # (24, 42)
    W = [[0] * rank for _ in range(p * n)]   # (12, 42)
    
    # Strassen 2x2x2x2 decomposition coefficients
    # For each Strassen term s, define contributions to U, V, W
    # U: (i, j_local) -> coefficient, where i in {0,1}, j_local in {0,1}
    # V: (j_local, k_local) -> coefficient, where j_local in {0,1}, k_local in {0,1}
    # W: (i, k_local) -> coefficient, where i in {0,1}, k_local in {0,1}
    
    # Strassen terms: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping (i, j_local) to coeff
    # v_coeffs: dict mapping (j_local, k_local) to coeff
    # w_coeffs: dict mapping (i, k_local) to coeff
    
    strassen = [
        # s=0: M1 = (m11+m22)(n11+n22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # s=1: M2 = (m21+m22)n11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # s=2: M3 = m11(n12-n22)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): -1}),
        # s=3: M4 = m22(n21-n11)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1, (1,1): -1}),
        # s=4: M5 = (m11+m12)n22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (1,1): 1}),
        # s=5: M6 = (m21-m11)(n11+n12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): 1}),
        # s=6: M7 = (m12-m22)(n21+n22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,1): 1, (0,0): -1}),
    ]
    
    for b in range(2):  # block: 0=left/top, 1=right/bottom
        for c in range(3):  # column block of the 2x6 result
            for s in range(7):
                col = (b * 3 + c) * 7 + s
                u_coeffs, v_coeffs, w_coeffs = strassen[s]
                
                # Fill U: rows are i*4 + (2*b + j_local) for i in {0,1}, j_local in {0,1}
                for (i, jl), coeff in u_coeffs.items():
                    row = i * m + (2 * b + jl)
                    U[row][col] += coeff
                
                # Fill V: rows are (2*b + jl) * 6 + (2*c + kl) for jl in {0,1}, kl in {0,1}
                for (jl, kl), coeff in v_coeffs.items():
                    row = (2 * b + jl) * p + (2 * c + kl)
                    V[row][col] += coeff
                
                # Fill W: rows are (2*c + kl) * 2 + i for i in {0,1}, kl in {0,1}
                for (i, kl), coeff in w_coeffs.items():
                    row = (2 * c + kl) * n + i
                    W[row][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
