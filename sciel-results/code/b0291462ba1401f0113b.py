"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44, W[k*n + i]

    # Define the 11 terms for a 3x2 by 2x2 multiplication
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
    
    terms = [
        # P1 = (A00 + A11)(B00 + B11)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # P2 = (A10 + A11)B00
        ([2, 3], [1, 1], [0], [1], [2], [1]),
        # P3 = A00(B10 - B11)
        ([0], [1], [2, 3], [1, -1], [1], [1]),
        # P4 = A11(B01 - B00)
        ([3], [1], [1, 0], [1, -1], [0, 1], [-1, 1]),
        # P5 = (A00 + A01)B11
        ([0, 1], [1, 1], [3], [1], [3], [1]),
        # P6 = (A10 - A00)(B00 + B01)
        ([2, 0], [1, -1], [0, 1], [1, 1], [0], [1]),
        # P7 = (A01 - A11)(B10 + B11)
        ([1, 3], [1, -1], [2, 3], [1, 1], [0, 2], [-1, 1]),
        # P8 = A20 B00
        ([4], [1], [0], [1], [4], [1]),
        # P9 = A20 B01
        ([4], [1], [1], [1], [5], [1]),
        # P10 = A21 B00
        ([5], [1], [0], [1], [4], [1]),
        # P11 = A21 B01
        ([5], [1], [1], [1], [5], [1]),
    ]

    # Define mappings for the 4 sets
    # Set 1: A00 B00 -> C00
    # Set 2: A01 B10 -> C00
    # Set 3: A00 B01 -> C01
    # Set 4: A01 B11 -> C01
    
    # Mapping functions: local (r, c) -> global index
    def map_A_set1(r, c): return r * 4 + c
    def map_B_set1(r, c): return r * 4 + c
    def map_C_set1(r, c): return c * 3 + r

    def map_A_set2(r, c): return r * 4 + (2 + c)
    def map_B_set2(r, c): return (2 + r) * 4 + c
    def map_C_set2(r, c): return c * 3 + r

    def map_A_set3(r, c): return r * 4 + c
    def map_B_set3(r, c): return r * 4 + (2 + c)
    def map_C_set3(r, c): return (2 + c) * 3 + r

    def map_A_set4(r, c): return r * 4 + (2 + c)
    def map_B_set4(r, c): return (2 + r) * 4 + (2 + c)
    def map_C_set4(r, c): return (2 + c) * 3 + r

    sets = [
        (map_A_set1, map_B_set1, map_C_set1),
        (map_A_set2, map_B_set2, map_C_set2),
        (map_A_set3, map_B_set3, map_C_set3),
        (map_A_set4, map_B_set4, map_C_set4),
    ]

    col = 0
    for map_A, map_B, map_C in sets:
        for term in terms:
            u_locals, u_vals, v_locals, v_vals, w_locals, w_vals = term
            
            for idx, val in zip(u_locals, u_vals):
                r, c = divmod(idx, 2)
                g_idx = map_A(r, c)
                U[g_idx][col] += val

            for idx, val in zip(v_locals, v_vals):
                r, c = divmod(idx, 2)
                g_idx = map_B(r, c)
                V[g_idx][col] += val

            for idx, val in zip(w_locals, w_vals):
                r, c = divmod(idx, 2)
                g_idx = map_C(r, c)
                W[g_idx][col] += val

            col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
