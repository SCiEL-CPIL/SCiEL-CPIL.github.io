"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen 2x2x2 coefficients
    u_coeffs = [
        [1, 0, 0, 1],  # P1: x0+x3
        [0, 0, 1, 1],  # P2: x2+x3
        [1, 0, 0, 0],  # P3: x0
        [0, 0, 0, 1],  # P4: x3
        [1, 1, 0, 0],  # P5: x0+x1
        [1, 0, 1, 0],  # P6: x2+x0
        [0, 1, 0, 1],  # P7: x3+x1
    ]
    v_coeffs = [
        [1, 0, 0, 1],  # P1: y0+y3
        [1, 0, 0, 0],  # P2: y0
        [0, 1, 1, 0],  # P3: y1+y2
        [0, 0, 1, 0],  # P4: y2
        [0, 0, 0, 1],  # P5: y3
        [1, 1, 0, 0],  # P6: y0+y1
        [0, 0, 1, 1],  # P7: y2+y3
    ]
    w_coeffs = [
        [(0, 1), (3, 1), (4, -1), (6, -1)],  # Z00
        [(2, 1), (4, 1)],                      # Z01
        [(1, 1), (3, 1)],                      # Z10
        [(0, 1), (1, -1), (2, 1), (5, -1)],   # Z11
    ]
    
    # Four Strassen instances
    instances = [
        {'x': [0, 1, 4, 5], 'y': [0, 1, 4, 5], 'z_to_w': [0, 1, 2, 3]},   # A1*B1
        {'x': [2, 3, 6, 7], 'y': [8, 9, 12, 13], 'z_to_w': [0, 1, 2, 3]}, # A2*B3
        {'x': [0, 1, 4, 5], 'y': [2, 3, 6, 7], 'z_to_w': [4, 5, 6, 7]},   # A1*B2
        {'x': [2, 3, 6, 7], 'y': [10, 11, 14, 15], 'z_to_w': [4, 5, 6, 7]},# A2*B4
    ]
    
    for inst_idx, inst in enumerate(instances):
        base = inst_idx * 7
        x_idx = inst['x']
        y_idx = inst['y']
        z_to_w = inst['z_to_w']
        
        for prod in range(7):
            col = base + prod
            for xi in range(4):
                coeff = u_coeffs[prod][xi]
                if coeff != 0:
                    U[x_idx[xi]][col] += coeff
            for yi in range(4):
                coeff = v_coeffs[prod][yi]
                if coeff != 0:
                    V[y_idx[yi]][col] += coeff
            for zi in range(4):
                for (pi, wc) in w_coeffs[zi]:
                    if pi == prod:
                        W[z_to_w[zi]][col] += wc
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
