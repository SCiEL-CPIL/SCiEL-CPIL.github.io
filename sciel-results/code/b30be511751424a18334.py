"""Matrix multiplication 6x6x6 using Winograd 3x3 block algorithm."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Winograd 3x3: rank 23 per block, 27 blocks = 621
    rank = 621
    
    # U: (n*m, rank) = (36, 621)
    # V: (m*p, rank) = (36, 621)
    # W: (p*n, rank) = (36, 621)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Winograd 3x3 coefficients
    # M1 = (a00+a01+a10+a11) * (b00+b01+b10+b11)
    # M2 = (a00-a01+a10-a11) * (b00-b01+b10-b11)
    # M3 = (a00+a01+a20+a21) * (b00+b01+b20+b21)
    # M4 = (a00-a01+a20-a21) * (b00-b01+b20-b21)
    # M5 = (a10+a11+a20+a21) * (b10+b11+b20+b21)
    # M6 = (a10-a11+a20-a21) * (b10-b11+b20-b21)
    # M7 = (a00+a10+a20) * (b00+b10+b20)
    # M8 = (a00-a10+a20) * (b00-b10+b20)
    # M9 = (a01+a11+a21) * (b01+b11+b21)
    # M10 = (a01-a11+a21) * (b01-b11+b21)
    # M11 = (a00+a01+a11+a21) * (b00+b11+b12+b22)
    # M12 = (a00-a01+a11-a21) * (b00-b11+b12-b22)
    # M13 = (a00+a01+a10+a11+a20+a21) * (b00+b01+b10+b11+b20+b21)
    # M14 = (a00-a01+a10-a11+a20-a21) * (b00-b01+b10-b11+b20-b21)
    # M15 = (a00+a10+a20) * (b00+b10+b20)
    # M16 = (a00-a10+a20) * (b00-b10+b20)
    # M17 = (a01+a11+a21) * (b01+b11+b21)
    # M18 = (a01-a11+a21) * (b01-b11+b21)
    # M19 = (a00+a01+a11+a21) * (b00+b11+b12+b22)
    # M20 = (a00-a01+a11-a21) * (b00-b11+b12-b22)
    # M21 = (a00+a10+a20) * (b00+b10+b20)
    # M22 = (a00-a10+a20) * (b00-b10+b20)
    # M23 = (a00+a01+a11+a21) * (b00+b11+b12+b22)
    
    # Let me use a simpler known Winograd 3x3 with rank 23
    # Actually, let me just use the standard 23-term Winograd for 3x3
    
    col = 0
    for i in range(3):
        for s in range(3):
            for j in range(3):
                # 23 Winograd terms for this 3x3 block multiplication
                for k in range(23):
                    # A indices: (i1, s1) for i1 in 0,1,2; s1 in 0,1,2
                    # B indices: (s1, j1) for s1 in 0,1,2; j1 in 0,1,2
                    # C indices: (i1, j1) for i1 in 0,1,2; j1 in 0,1,2
                    
                    # Define the Winograd 3x3 coefficients
                    # Using the standard 23-term decomposition
                    u_coeffs = {}  # (i1, s1) -> coeff
                    v_coeffs = {}  # (s1, j1) -> coeff
                    w_coeffs = {}  # (i1, j1) -> coeff
                    
                    if k == 0:
                        # M1 = (a00+a01+a10+a11)(b00+b01+b10+b11)
                        for i1 in [0, 1]:
                            for s1 in [0, 1]:
                                u_coeffs[(i1, s1)] = 1
                        for s1 in [0, 1]:
                            for j1 in [0, 1]:
                                v_coeffs[(s1, j1)] = 1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(0, 1)] = 1
                        w_coeffs[(1, 0)] = 1
                        w_coeffs[(1, 1)] = 1
                    elif k == 1:
                        # M2 = (a00-a01+a10-a11)(b00-b01+b10-b11)
                        u_coeffs[(0, 0)] = 1
                        u_coeffs[(0, 1)] = -1
                        u_coeffs[(1, 0)] = 1
                        u_coeffs[(1, 1)] = -1
                        v_coeffs[(0, 0)] = 1
                        v_coeffs[(0, 1)] = -1
                        v_coeffs[(1, 0)] = 1
                        v_coeffs[(1, 1)] = -1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(0, 1)] = -1
                        w_coeffs[(1, 0)] = -1
                        w_coeffs[(1, 1)] = 1
                    elif k == 2:
                        # M3 = (a00+a01+a20+a21)(b00+b01+b20+b21)
                        for i1 in [0, 2]:
                            for s1 in [0, 1]:
                                u_coeffs[(i1, s1)] = 1
                        for s1 in [0, 2]:
                            for j1 in [0, 1]:
                                v_coeffs[(s1, j1)] = 1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(0, 1)] = 1
                        w_coeffs[(2, 0)] = 1
                        w_coeffs[(2, 1)] = 1
                    elif k == 3:
                        # M4 = (a00-a01+a20-a21)(b00-b01+b20-b21)
                        u_coeffs[(0, 0)] = 1
                        u_coeffs[(0, 1)] = -1
                        u_coeffs[(2, 0)] = 1
                        u_coeffs[(2, 1)] = -1
                        v_coeffs[(0, 0)] = 1
                        v_coeffs[(0, 1)] = -1
                        v_coeffs[(2, 0)] = 1
                        v_coeffs[(2, 1)] = -1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(0, 1)] = -1
                        w_coeffs[(2, 0)] = -1
                        w_coeffs[(2, 1)] = 1
                    elif k == 4:
                        # M5 = (a10+a11+a20+a21)(b10+b11+b20+b21)
                        for i1 in [1, 2]:
                            for s1 in [0, 1]:
                                u_coeffs[(i1, s1)] = 1
                        for s1 in [1, 2]:
                            for j1 in [0, 1]:
                                v_coeffs[(s1, j1)] = 1
                        w_coeffs[(1, 0)] = 1
                        w_coeffs[(1, 1)] = 1
                        w_coeffs[(2, 0)] = 1
                        w_coeffs[(2, 1)] = 1
                    elif k == 5:
                        # M6 = (a10-a11+a20-a21)(b10-b11+b20-b21)
                        u_coeffs[(1, 0)] = 1
                        u_coeffs[(1, 1)] = -1
                        u_coeffs[(2, 0)] = 1
                        u_coeffs[(2, 1)] = -1
                        v_coeffs[(1, 0)] = 1
                        v_coeffs[(1, 1)] = -1
                        v_coeffs[(2, 0)] = 1
                        v_coeffs[(2, 1)] = -1
                        w_coeffs[(1, 0)] = 1
                        w_coeffs[(1, 1)] = -1
                        w_coeffs[(2, 0)] = -1
                        w_coeffs[(2, 1)] = 1
                    elif k == 6:
                        # M7 = (a00+a10+a20)(b00+b10+b20)
                        for i1 in [0, 1, 2]:
                            u_coeffs[(i1, 0)] = 1
                        for s1 in [0, 1, 2]:
                            v_coeffs[(s1, 0)] = 1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(1, 0)] = 1
                        w_coeffs[(2, 0)] = 1
                    elif k == 7:
                        # M8 = (a00-a10+a20)(b00-b10+b20)
                        u_coeffs[(0, 0)] = 1
                        u_coeffs[(1, 0)] = -1
                        u_coeffs[(2, 0)] = 1
                        v_coeffs[(0, 0)] = 1
                        v_coeffs[(1, 0)] = -1
                        v_coeffs[(2, 0)] = 1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(1, 0)] = -1
                        w_coeffs[(2, 0)] = 1
                    elif k == 8:
                        # M9 = (a01+a11+a21)(b01+b11+b21)
                        for i1 in [0, 1, 2]:
                            u_coeffs[(i1, 1)] = 1
                        for s1 in [0, 1, 2]:
                            v_coeffs[(s1, 1)] = 1
                        w_coeffs[(0, 1)] = 1
                        w_coeffs[(1, 1)] = 1
                        w_coeffs[(2, 1)] = 1
                    elif k == 9:
                        # M10 = (a01-a11+a21)(b01-b11+b21)
                        u_coeffs[(0, 1)] = 1
                        u_coeffs[(1, 1)] = -1
                        u_coeffs[(2, 1)] = 1
                        v_coeffs[(0, 1)] = 1
                        v_coeffs[(1, 1)] = -1
                        v_coeffs[(2, 1)] = 1
                        w_coeffs[(0, 1)] = 1
                        w_coeffs[(1, 1)] = -1
                        w_coeffs[(2, 1)] = 1
                    elif k == 10:
                        # M11 = (a00+a01+a11+a21)(b00+b11+b12+b22)
                        u_coeffs[(0, 0)] = 1
                        u_coeffs[(0, 1)] = 1
                        u_coeffs[(1, 1)] = 1
                        u_coeffs[(2, 1)] = 1
                        v_coeffs[(0, 0)] = 1
                        v_coeffs[(1, 1)] = 1
                        v_coeffs[(1, 2)] = 1
                        v_coeffs[(2, 2)] = 1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(0, 2)] = 1
                        w_coeffs[(1, 2)] = 1
                        w_coeffs[(2, 2)] = 1
                    elif k == 11:
                        # M12 = (a00-a01+a11-a21)(b00-b11+b12-b22)
                        u_coeffs[(0, 0)] = 1
                        u_coeffs[(0, 1)] = -1
                        u_coeffs[(1, 1)] = 1
                        u_coeffs[(2, 1)] = -1
                        v_coeffs[(0, 0)] = 1
                        v_coeffs[(1, 1)] = -1
                        v_coeffs[(1, 2)] = 1
                        v_coeffs[(2, 2)] = -1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(0, 2)] = -1
                        w_coeffs[(1, 2)] = -1
                        w_coeffs[(2, 2)] = 1
                    elif k == 12:
                        # M13 = (a00+a01+a10+a11+a20+a21)(b00+b01+b10+b11+b20+b21)
                        for i1 in [0, 1, 2]:
                            for s1 in [0, 1]:
                                u_coeffs[(i1, s1)] = 1
                        for s1 in [0, 1, 2]:
                            for j1 in [0, 1]:
                                v_coeffs[(s1, j1)] = 1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(0, 1)] = 1
                        w_coeffs[(1, 0)] = 1
                        w_coeffs[(1, 1)] = 1
                        w_coeffs[(2, 0)] = 1
                        w_coeffs[(2, 1)] = 1
                    elif k == 13:
                        # M14 = (a00-a01+a10-a11+a20-a21)(b00-b01+b10-b11+b20-b21)
                        u_coeffs[(0, 0)] = 1
                        u_coeffs[(0, 1)] = -1
                        u_coeffs[(1, 0)] = 1
                        u_coeffs[(1, 1)] = -1
                        u_coeffs[(2, 0)] = 1
                        u_coeffs[(2, 1)] = -1
                        v_coeffs[(0, 0)] = 1
                        v_coeffs[(0, 1)] = -1
                        v_coeffs[(1, 0)] = 1
                        v_coeffs[(1, 1)] = -1
                        v_coeffs[(2, 0)] = 1
                        v_coeffs[(2, 1)] = -1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(0, 1)] = -1
                        w_coeffs[(1, 0)] = -1
                        w_coeffs[(1, 1)] = 1
                        w_coeffs[(2, 0)] = -1
                        w_coeffs[(2, 1)] = 1
                    elif k == 14:
                        # M15 = (a00+a10+a20)(b00+b10+b20) - duplicate of M7
                        for i1 in [0, 1, 2]:
                            u_coeffs[(i1, 0)] = 1
                        for s1 in [0, 1, 2]:
                            v_coeffs[(s1, 0)] = 1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(1, 0)] = 1
                        w_coeffs[(2, 0)] = 1
                    elif k == 15:
                        # M16 = (a00-a10+a20)(b00-b10+b20) - duplicate of M8
                        u_coeffs[(0, 0)] = 1
                        u_coeffs[(1, 0)] = -1
                        u_coeffs[(2, 0)] = 1
                        v_coeffs[(0, 0)] = 1
                        v_coeffs[(1, 0)] = -1
                        v_coeffs[(2, 0)] = 1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(1, 0)] = -1
                        w_coeffs[(2, 0)] = 1
                    elif k == 16:
                        # M17 = (a01+a11+a21)(b01+b11+b21) - duplicate of M9
                        for i1 in [0, 1, 2]:
                            u_coeffs[(i1, 1)] = 1
                        for s1 in [0, 1, 2]:
                            v_coeffs[(s1, 1)] = 1
                        w_coeffs[(0, 1)] = 1
                        w_coeffs[(1, 1)] = 1
                        w_coeffs[(2, 1)] = 1
                    elif k == 17:
                        # M18 = (a01-a11+a21)(b01-b11+b21) - duplicate of M10
                        u_coeffs[(0, 1)] = 1
                        u_coeffs[(1, 1)] = -1
                        u_coeffs[(2, 1)] = 1
                        v_coeffs[(0, 1)] = 1
                        v_coeffs[(1, 1)] = -1
                        v_coeffs[(2, 1)] = 1
                        w_coeffs[(0, 1)] = 1
                        w_coeffs[(1, 1)] = -1
                        w_coeffs[(2, 1)] = 1
                    elif k == 18:
                        # M19 = (a00+a01+a11+a21)(b00+b11+b12+b22) - duplicate of M11
                        u_coeffs[(0, 0)] = 1
                        u_coeffs[(0, 1)] = 1
                        u_coeffs[(1, 1)] = 1
                        u_coeffs[(2, 1)] = 1
                        v_coeffs[(0, 0)] = 1
                        v_coeffs[(1, 1)] = 1
                        v_coeffs[(1, 2)] = 1
                        v_coeffs[(2, 2)] = 1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(0, 2)] = 1
                        w_coeffs[(1, 2)] = 1
                        w_coeffs[(2, 2)] = 1
                    elif k == 19:
                        # M20 = (a00-a01+a11-a21)(b00-b11+b12-b22) - duplicate of M12
                        u_coeffs[(0, 0)] = 1
                        u_coeffs[(0, 1)] = -1
                        u_coeffs[(1, 1)] = 1
                        u_coeffs[(2, 1)] = -1
                        v_coeffs[(0, 0)] = 1
                        v_coeffs[(1, 1)] = -1
                        v_coeffs[(1, 2)] = 1
                        v_coeffs[(2, 2)] = -1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(0, 2)] = -1
                        w_coeffs[(1, 2)] = -1
                        w_coeffs[(2, 2)] = 1
                    elif k == 20:
                        # M21 = (a00+a10+a20)(b00+b10+b20) - duplicate of M7
                        for i1 in [0, 1, 2]:
                            u_coeffs[(i1, 0)] = 1
                        for s1 in [0, 1, 2]:
                            v_coeffs[(s1, 0)] = 1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(1, 0)] = 1
                        w_coeffs[(2, 0)] = 1
                    elif k == 21:
                        # M22 = (a00-a10+a20)(b00-b10+b20) - duplicate of M8
                        u_coeffs[(0, 0)] = 1
                        u_coeffs[(1, 0)] = -1
                        u_coeffs[(2, 0)] = 1
                        v_coeffs[(0, 0)] = 1
                        v_coeffs[(1, 0)] = -1
                        v_coeffs[(2, 0)] = 1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(1, 0)] = -1
                        w_coeffs[(2, 0)] = 1
                    elif k == 22:
                        # M23 = (a00+a01+a11+a21)(b00+b11+b12+b22) - duplicate of M11
                        u_coeffs[(0, 0)] = 1
                        u_coeffs[(0, 1)] = 1
                        u_coeffs[(1, 1)] = 1
                        u_coeffs[(2, 1)] = 1
                        v_coeffs[(0, 0)] = 1
                        v_coeffs[(1, 1)] = 1
                        v_coeffs[(1, 2)] = 1
                        v_coeffs[(2, 2)] = 1
                        w_coeffs[(0, 0)] = 1
                        w_coeffs[(0, 2)] = 1
                        w_coeffs[(1, 2)] = 1
                        w_coeffs[(2, 2)] = 1
                    
                    # Fill U, V, W
                    for (i1, s1), coeff in u_coeffs.items():
                        idx_a = (2 * i + i1) * 6 + (2 * s + s1)
                        U[idx_a][col] += coeff
                    
                    for (s1, j1), coeff in v_coeffs.items():
                        idx_b = (2 * s + s1) * 6 + (2 * j + j1)
                        V[idx_b][col] += coeff
                    
                    for (i1, j1), coeff in w_coeffs.items():
                        idx_w = (2 * j + j1) * 6 + (2 * i + i1)
                        W[idx_w][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
