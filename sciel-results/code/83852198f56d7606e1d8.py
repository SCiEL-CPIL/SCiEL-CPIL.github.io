import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: Strassen for 4x4x4 (first 4 columns of B)
    # We generate 49 terms.
    strassen_terms_2x2 = [
        # (u_indices, v_indices, w_indices)
        # u, v, w are lists of (index, coeff)
        # 2x2 indices: 00=0, 01=1, 10=2, 11=3
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),      # P1: (a+d)(e+h) -> C00, C11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)]),              # P2: (c+d)e -> C10, C11
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),             # P3: a(f-h) -> C01, C11
        ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (3, 1)]),             # P4: d(g-e) -> C10, C11
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, -1)]),             # P5: (a+b)h -> C01, C11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(1, 1), (3, 1)]),     # P6: (c-a)(e+f) -> C01, C11
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)])      # P7: (b-d)(g+h) -> C00, C11
    ]
    
    # Note: The w_indices above need to be verified against Strassen formulas.
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's re-verify w for each P:
    # P1: C00 (+1), C11 (+1). w=[(0,1), (3,1)]
    # P2: C10 (+1), C11 (-1). w=[(2,1), (3,-1)]
    # P3: C01 (+1), C11 (+1). w=[(1,1), (3,1)]
    # P4: C10 (+1), C11 (+1). w=[(2,1), (3,1)]
    # P5: C01 (+1), C11 (-1). w=[(1,1), (3,-1)]
    # P6: C11 (+1). w=[(3,1)]
    # P7: C00 (+1), C11 (-1). w=[(0,1), (3,-1)]
    
    # Corrected Strassen 2x2 terms:
    s_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),      # P1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),             # P2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),             # P3
        ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (3, 1)]),             # P4
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, -1)]),             # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),             # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)])     # P7
    ]

    # Generate 4x4 Strassen terms
    # Top level linear forms for A, B, C blocks
    # A blocks: A00(0), A01(1), A10(2), A11(3)
    # B blocks: B00(0), B01(1), B10(2), B11(3)
    # C blocks: C00(0), C01(1), C10(2), C11(3)
    
    top_A = [
        [(0, 1), (3, 1)],  # P1: A00+A11
        [(2, 1), (3, 1)],  # P2: A10+A11
        [(0, 1)],          # P3: A00
        [(3, 1)],          # P4: A11
        [(0, 1), (1, 1)],  # P5: A00+A01
        [(2, 1), (0, -1)], # P6: A10-A00
        [(1, 1), (3, -1)]  # P7: A01-A11
    ]
    top_B = [
        [(0, 1), (3, 1)],  # P1: B00+B11
        [(0, 1)],          # P2: B00
        [(1, 1), (3, -1)], # P3: B01-B11
        [(2, 1), (0, -1)], # P4: B10-B00
        [(3, 1)],          # P5: B11
        [(0, 1), (1, 1)],  # P6: B00+B01
        [(2, 1), (3, 1)]   # P7: B10+B11
    ]
    top_C = [
        [(0, 1), (3, 1)],  # P1: C00+C11
        [(2, 1), (3, -1)], # P2: C10-C11
        [(1, 1), (3, 1)],  # P3: C01+C11
        [(2, 1), (3, 1)],  # P4: C10+C11
        [(1, 1), (3, -1)], # P5: C01-C11
        [(3, 1)],          # P6: C11
        [(0, 1), (3, -1)]  # P7: C00-C11
    ]

    def compose_forms(block_form, base_form, block_size):
        # block_form: list of (block_idx, coeff)
        # base_form: list of (local_idx, coeff)
        # block_size: 2 for 2x2 blocks
        # Returns list of (global_idx, coeff)
        result = {}
        for b_idx, b_coeff in block_form:
            for l_idx, l_coeff in base_form:
                # Global index = b_idx * (block_size*block_size) + l_idx
                g_idx = b_idx * (block_size * block_size) + l_idx
                coeff = b_coeff * l_coeff
                if g_idx in result:
                    result[g_idx] += coeff
                else:
                    result[g_idx] = coeff
        return [(k, v) for k, v in result.items() if v != 0]

    terms_4x4 = []
    for k in range(7):
        for s in s_terms:
            u_s, v_s, w_s = s
            u_global = compose_forms(top_A[k], u_s, 2)
            v_global = compose_forms(top_B[k], v_s, 2)
            w_global = compose_forms(top_C[k], w_s, 2)
            terms_4x4.append((u_global, v_global, w_global))
            
    # Part 2: Schoolbook for 4x4x1 (5th column)
    terms_1x1 = []
    for i in range(4):
        for j in range(4):
            u = [(i * 4 + j, 1)]
            v = [(j * 5 + 4, 1)] # B[j, 4]
            w = [(4 * 4 + i, 1)] # C[i, 4] -> W index k*n+i = 4*4+i
            terms_1x1.append((u, v, w))

    all_terms = terms_4x4 + terms_1x1
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(all_terms):
        for idx, coeff in u:
            U[idx][r] = coeff
        for idx, coeff in v:
            V[idx][r] = coeff
        for idx, coeff in w:
            W[idx][r] = coeff
            
    return {"U": U, "V": V, "W": W}
