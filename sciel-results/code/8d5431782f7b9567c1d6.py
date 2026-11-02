"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition coefficients
    # U: 4 coeffs for A (a,b,c,d)
    # V: 4 coeffs for B (e,f,g,h)
    # W: 4 coeffs for C (c11, c12, c21, c22)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, -1, 0, 1])
    ]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # A is 2x4, split into A1 (cols 0-1), A2 (cols 2-3)
    # B is 4x6, split into B11, B12, B13 (rows 0-1) and B21, B22, B23 (rows 2-3)
    # C is 2x6, split into C1, C2, C3 (cols 0-1, 2-3, 4-5)
    
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    # C3 = A1 B13 + A2 B23
    
    # Indices in U (n*m = 8):
    # A1: rows 0-1, cols 0-1 -> U indices 0,1,2,3
    # A2: rows 0-1, cols 2-3 -> U indices 4,5,6,7
    
    # Indices in V (m*p = 24):
    # B11: rows 0-1, cols 0-1 -> V indices 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> V indices 2,3,6,7
    # B13: rows 0-1, cols 4-5 -> V indices 8,9,12,13
    # B21: rows 2-3, cols 0-1 -> V indices 8,9,12,13? No.
    # V index = row * p + col.
    # B11: r=0,1 c=0,1 -> 0,1, 6,7
    # B12: r=0,1 c=2,3 -> 2,3, 8,9
    # B13: r=0,1 c=4,5 -> 4,5, 10,11
    # B21: r=2,3 c=0,1 -> 12,13, 18,19
    # B22: r=2,3 c=2,3 -> 14,15, 20,21
    # B23: r=2,3 c=4,5 -> 16,17, 22,23
    
    # Indices in W (p*n = 12):
    # W index = k * n + i.
    # C1: k=0,1 i=0,1 -> 0,1, 2,3
    # C2: k=2,3 i=0,1 -> 4,5, 6,7
    # C3: k=4,5 i=0,1 -> 8,9, 10,11
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, u_base, v_base, w_base):
        nonlocal col
        for r in range(7):
            u, v, w = strassen_terms[r]
            for i in range(4):
                if u[i] != 0:
                    U[u_base + i][col] += u[i]
                if v[i] != 0:
                    V[v_base + i][col] += v[i]
                if w[i] != 0:
                    W[w_base + i][col] += w[i]
            col += 1

    # C1 = A1 B11 + A2 B21
    # Term 1: A1 B11
    # A1 base 0, B11 base 0, C1 base 0
    add_term(None, None, None, 0, 0, 0)
    # Term 2: A2 B21
    # A2 base 4, B21 base 12, C1 base 0
    add_term(None, None, None, 4, 12, 0)
    
    # C2 = A1 B12 + A2 B22
    # Term 3: A1 B12
    # A1 base 0, B12 base 2, C2 base 4
    add_term(None, None, None, 0, 2, 4)
    # Term 4: A2 B22
    # A2 base 4, B22 base 14, C2 base 4
    add_term(None, None, None, 4, 14, 4)
    
    # C3 = A1 B13 + A2 B23
    # Term 5: A1 B13
    # A1 base 0, B13 base 4, C3 base 8
    add_term(None, None, None, 0, 4, 8)
    # Term 6: A2 B23
    # A2 base 4, B23 base 16, C3 base 8
    add_term(None, None, None, 4, 16, 8)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
