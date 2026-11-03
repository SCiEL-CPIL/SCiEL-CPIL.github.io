import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Rank 42 via 6 blocks of Strassen 2x2x2
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms definitions
    # Each term is a list of (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
    # M2 = (c+d)e -> C10 += 1
    # M3 = a(f-h) -> C01 += 1
    # M4 = d(g-e) -> C00 += 1, C10 += 1
    # M5 = (a+b)h -> C00 -= 1, C01 += 1
    # M6 = (c-a)(e+f) -> C11 += 1
    # M7 = (b-d)(g+h) -> C00 += 1, C11 -= 1
    
    strassen_terms = [
        # (u_idx, u_val, v_idx, v_val, w_idx, w_val)
        # M1
        [(0, 1), (3, 1), (0, 1), (3, 1), (0, 1), (3, 1)],
        # M2
        [(2, 1), (3, 1), (0, 1), (None, None), (2, 1), (None, None)], # Note: structure needs to be consistent
    ]
    
    # It is easier to just hardcode the loops for the 7 terms inside the block loop
    # to avoid complex data structures.
    
    def add_term(u_offsets, v_offsets, w_offsets):
        nonlocal col
        for idx, val in u_offsets:
            U[idx][col] += val
        for idx, val in v_offsets:
            V[idx][col] += val
        for idx, val in w_offsets:
            W[idx][col] += val
        col += 1

    for b_k in range(3):  # 3 blocks in p dimension (size 2 each)
        for b_j in range(2):  # 2 blocks in m dimension (size 2 each)
            # Base offsets
            # i is 0..1
            # j_local is 0..1, j_global = 2*b_j + j_local
            # k_local is 0..1, k_global = 2*b_k + k_local
            
            # Helper to get global indices
            def u_idx(i, jl): return i * m + (2 * b_j + jl)
            def v_idx(jl, kl): return (2 * b_j + jl) * p + (2 * b_k + kl)
            def w_idx(kl, i): return (2 * b_k + kl) * n + i
            
            # Term 1: (a+d)(e+h)
            # a: u(0,0), d: u(1,1)
            # e: v(0,0), h: v(1,1)
            # c00: w(0,0), c11: w(1,1)
            add_term([(u_idx(0,0), 1), (u_idx(1,1), 1)],
                     [(v_idx(0,0), 1), (v_idx(1,1), 1)],
                     [(w_idx(0,0), 1), (w_idx(1,1), 1)])
                     
            # Term 2: (c+d)e
            # c: u(1,0), d: u(1,1)
            # e: v(0,0)
            # c10: w(0,1) -> wait, w(kl, i). c10 is i=1, k=0. So w(0,1).
            add_term([(u_idx(1,0), 1), (u_idx(1,1), 1)],
                     [(v_idx(0,0), 1)],
                     [(w_idx(0,1), 1)])
                     
            # Term 3: a(f-h)
            # a: u(0,0)
            # f: v(0,1), h: v(1,1)
            # c01: w(1,0) -> i=0, k=1. w(1,0).
            add_term([(u_idx(0,0), 1)],
                     [(v_idx(0,1), 1), (v_idx(1,1), -1)],
                     [(w_idx(1,0), 1)])
                     
            # Term 4: d(g-e)
            # d: u(1,1)
            # g: v(1,0), e: v(0,0)
            # c00: w(0,0), c10: w(0,1)
            add_term([(u_idx(1,1), 1)],
                     [(v_idx(1,0), 1), (v_idx(0,0), -1)],
                     [(w_idx(0,0), 1), (w_idx(0,1), 1)])
                     
            # Term 5: (a+b)h
            # a: u(0,0), b: u(0,1)
            # h: v(1,1)
            # c00: w(0,0) -1, c01: w(1,0) +1
            add_term([(u_idx(0,0), 1), (u_idx(0,1), 1)],
                     [(v_idx(1,1), 1)],
                     [(w_idx(0,0), -1), (w_idx(1,0), 1)])
                     
            # Term 6: (c-a)(e+f)
            # c: u(1,0), a: u(0,0)
            # e: v(0,0), f: v(0,1)
            # c11: w(1,1)
            add_term([(u_idx(1,0), 1), (u_idx(0,0), -1)],
                     [(v_idx(0,0), 1), (v_idx(0,1), 1)],
                     [(w_idx(1,1), 1)])
                     
            # Term 7: (b-d)(g+h)
            # b: u(0,1), d: u(1,1)
            # g: v(1,0), h: v(1,1)
            # c00: w(0,0) +1, c11: w(1,1) -1
            add_term([(u_idx(0,1), 1), (u_idx(1,1), -1)],
                     [(v_idx(1,0), 1), (v_idx(1,1), 1)],
                     [(w_idx(0,0), 1), (w_idx(1,1), -1)])
                     
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
