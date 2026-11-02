DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * (p // 2)
    num_pairs = p // 2
    rank = 7 * num_pairs
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # Products P1..P7
    # U coeffs (a,b,c,d)
    # V coeffs (e,f,g,h) -> (B0k1, B0k2, B1k1, B1k2)
    # W coeffs (C0k1, C1k1, C0k2, C1k2)
    
    # Note: My V mapping above was e=B0k1, f=B0k2, g=B1k1, h=B1k2
    # Let's stick to this order:
    # e = B[0, k1]
    # f = B[0, k2]
    # g = B[1, k1]
    # h = B[1, k2]
    
    # P1: (a+d)(e+h)
    # U: a=1, d=1
    # V: e=1, h=1
    # W: C0k1=1, C1k2=1
    # P2: (c+d)e
    # U: c=1, d=1
    # V: e=1
    # W: C1k1=1, C1k2=-1
    # P3: a(f-h)
    # U: a=1
    # V: f=1, h=-1
    # W: C0k2=1, C1k2=1
    # P4: d(g-e)
    # U: d=1
    # V: g=1, e=-1
    # W: C0k1=-1, C1k1=1
    # P5: (a+b)h
    # U: a=1, b=1
    # V: h=1
    # W: C0k1=-1, C0k2=1
    # P6: (c-a)(e+f)
    # U: c=1, a=-1
    # V: e=1, f=1
    # W: C1k2=1
    # P7: (b-d)(g+h)
    # U: b=1, d=-1
    # V: g=1, h=1
    # W: C0k1=1

    # Define the 7 products
    # Each product is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {index: val} for A
    # v_coeffs: dict {index: val} for B
    # w_coeffs: dict {index: val} for C
    
    products = [
        # P1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # e=0, h=3, C0k1=0, C1k2=3
        # P2
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}), # e=0, C1k1=1, C1k2=3
        # P3
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}), # f=1, h=3, C0k2=2, C1k2=3
        # P4
        ({3: 1}, {2: 1, 0: -1}, {0: -1, 1: 1}), # g=2, e=0, C0k1=0, C1k1=1
        # P5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}), # h=3, C0k1=0, C0k2=2
        # P6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # e=0, f=1, C1k2=3
        # P7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}), # g=2, h=3, C0k1=0
    ]
    
    # Map local indices to global indices
    # A: 0->00, 1->01, 2->10, 3->11
    # B local: 0->B0k1, 1->B0k2, 2->B1k1, 3->B1k2
    # C local: 0->C0k1, 1->C1k1, 2->C0k2, 3->C1k2
    
    col = 0
    for r in range(num_pairs):
        k1 = 2 * r
        k2 = 2 * r + 1
        
        # Global B indices
        # B[0, k1] -> 0 * p + k1 = k1
        # B[0, k2] -> k2
        # B[1, k1] -> 1 * p + k1 = p + k1
        # B[1, k2] -> p + k2
        b_map = {
            0: k1,
            1: k2,
            2: p + k1,
            3: p + k2
        }
        
        # Global C indices
        # C[0, k1] -> k1 * n + 0 = 2 * k1
        # C[1, k1] -> 2 * k1 + 1
        # C[0, k2] -> 2 * k2
        # C[1, k2] -> 2 * k2 + 1
        c_map = {
            0: 2 * k1,
            1: 2 * k1 + 1,
            2: 2 * k2,
            3: 2 * k2 + 1
        }
        
        for u_coeffs, v_coeffs, w_coeffs in products:
            # U
            for idx, val in u_coeffs.items():
                U[idx][col] = val
            # V
            for idx, val in v_coeffs.items():
                V[b_map[idx]][col] = val
            # W
            for idx, val in w_coeffs.items():
                W[c_map[idx]][col] = val
            col += 1
            
    return {"U": U, "V": V, "W": W}
