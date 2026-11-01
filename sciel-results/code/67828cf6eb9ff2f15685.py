"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    # Split p into 5 blocks of 2 and 1 block of 1
    # Rank = 5 * 7 + 4 = 39
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        # u_coeffs: dict {global_u_idx: val}
        # v_coeffs: dict {global_v_idx: val}
        # w_coeffs: dict {global_w_idx: val}
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1

    # Strassen coefficients for 2x2x2
    # Local indices:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)  <- Note: W index is k*n+i
    #    C_00 (k=0,i=0) -> 0
    #    C_10 (k=0,i=1) -> 1
    #    C_01 (k=1,i=0) -> 2
    #    C_11 (k=1,i=1) -> 3
    
    strassen_terms = [
        # 1: (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # 2: (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {1: 1}),
        # 3: a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {2: 1}),
        # 4: d(g-f)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # 5: (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # 6: (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # 7: (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]

    # Process 5 blocks of 2x2x2
    for b in range(5):
        k_start = b * 2
        # Global indices mapping
        # U: i in [0,2), j in [0,2)
        # V: j in [0,2), k in [k_start, k_start+2)
        # W: k in [k_start, k_start+2), i in [0,2)
        
        for u_loc, v_loc, w_loc in strassen_terms:
            u_global = {}
            for li, val in u_loc.items():
                # li: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                i = li // 2
                j = li % 2
                # Global U index: i * m + j
                u_global[i * m + j] = val
                
            v_global = {}
            for li, val in v_loc.items():
                # li: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                j = li // 2
                k = li % 2
                # Global V index: j * p + (k_start + k)
                v_global[j * p + (k_start + k)] = val
                
            w_global = {}
            for li, val in w_loc.items():
                # li: 0->(0,0), 1->(1,0), 2->(0,1), 3->(1,1)
                # Wait, local W indices in strassen_terms:
                # 0: C_00 (i=0, k=0)
                # 1: C_10 (i=1, k=0)
                # 2: C_01 (i=0, k=1)
                # 3: C_11 (i=1, k=1)
                if li == 0:
                    i, k = 0, 0
                elif li == 1:
                    i, k = 1, 0
                elif li == 2:
                    i, k = 0, 1
                else:
                    i, k = 1, 1
                # Global W index: k_global * n + i
                k_global = k_start + k
                w_global[k_global * n + i] = val
            
            add_term(u_global, v_global, w_global)

    # Process 1 block of 2x2x1
    k_start = 10
    # Terms:
    # 1: A_00 * B_0,10 -> C_0,10
    # 2: A_01 * B_1,10 -> C_0,10
    # 3: A_10 * B_0,10 -> C_1,10
    # 4: A_11 * B_1,10 -> C_1,10
    
    # 1
    add_term({0: 1}, {0 * p + k_start: 1}, {k_start * n + 0: 1})
    # 2
    add_term({1: 1}, {1 * p + k_start: 1}, {k_start * n + 0: 1})
    # 3
    add_term({2: 1}, {0 * p + k_start: 1}, {k_start * n + 1: 1})
    # 4
    add_term({3: 1}, {1 * p + k_start: 1}, {k_start * n + 1: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
