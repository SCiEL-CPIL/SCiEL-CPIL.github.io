DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms for 2x2x2: (U_coeffs, V_local_coeffs, W_local_coeffs)
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V local: 0=B00, 1=B01, 2=B10, 3=B11
    # W local: 0=C00, 1=C01, 2=C10, 3=C11
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1},       {2: 1, 3: -1}),       # M2
        ({0: 1},        {1: 1, 3: -1}, {1: 1, 3: 1}),       # M3
        ({3: 1},        {2: 1, 0: -1}, {0: 1, 2: 1}),       # M4
        ({0: 1, 1: 1},  {3: 1},       {0: -1, 1: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),              # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),              # M7
    ]

    for b in range(num_blocks):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = b * 7 + t_idx
            v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
            w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]

            for idx, val in u_dict.items():
                U[idx][col] += val
            for l_idx, val in v_dict.items():
                V[v_offsets[l_idx]][col] += val
            for l_idx, val in w_dict.items():
                W[w_offsets[l_idx]][col] += val

    return {"U": U, "V": V, "W": W}
