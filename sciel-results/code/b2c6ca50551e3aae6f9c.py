import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Initialize
    U = []
    V = []
    W = []
    
    def add_term(u_vals, v_vals, w_vals):
        # u_vals: dict {index: coeff}
        # v_vals: dict {index: coeff}
        # w_vals: dict {index: coeff}
        u_row = [0.0] * (n * m)
        v_row = [0.0] * (m * p)
        w_row = [0.0] * (p * n)
        
        for idx, val in u_vals.items():
            u_row[idx] = val
        for idx, val in v_vals.items():
            v_row[idx] = val
        for idx, val in w_vals.items():
            w_row[idx] = val
            
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)

    # Strassen terms for 2x2x2 block (u in 0,1; v in 0,1; w in 0,1)
    # Indices:
    # A: u*5+v
    # B: v*5+w
    # C: w*3+u
    
    # M1: (a+d)(e+h)
    add_term(
        {0: 1, 6: 1}, # a, d
        {0: 1, 6: 1}, # e, h
        {0: 1, 4: 1}  # C00, C11
    )
    
    # M2: (c+d)e
    add_term(
        {5: 1, 6: 1}, # c, d
        {0: 1},       # e
        {1: 1, 4: -1} # C10, C11
    )
    
    # M3: a(f-h)
    add_term(
        {0: 1},       # a
        {1: 1, 6: -1}, # f, h
        {3: 1, 4: 1}  # C01, C11
    )
    
    # M4: d(g-e)
    add_term(
        {6: 1},       # d
        {5: 1, 0: -1}, # g, e
        {0: 1, 1: 1}  # C00, C10
    )
    
    # M5: (a+b)h
    add_term(
        {0: 1, 1: 1}, # a, b
        {6: 1},       # h
        {0: -1, 3: 1} # C00, C01
    )
    
    # M6: (c-a)(e+f)
    add_term(
        {5: 1, 0: -1}, # c, a
        {0: 1, 1: 1},  # e, f
        {4: 1}         # C11
    )
    
    # M7: (b-d)(g+h)
    add_term(
        {1: 1, 6: -1}, # b, d
        {5: 1, 6: 1},  # g, h
        {0: 1}         # C00
    )

    # Schoolbook terms for all other (u, v, w)
    for u in range(n):
        for v in range(m):
            for w in range(p):
                # Skip the 2x2x2 block handled by Strassen
                if u in [0, 1] and v in [0, 1] and w in [0, 1]:
                    continue
                u_idx = u * m + v
                v_idx = v * p + w
                w_idx = w * n + u
                add_term(
                    {u_idx: 1},
                    {v_idx: 1},
                    {w_idx: 1}
                )

    return {"U": U, "V": V, "W": W}
