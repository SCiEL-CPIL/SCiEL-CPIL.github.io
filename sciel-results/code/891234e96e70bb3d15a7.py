"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # Part 1: A_1 (2x1) * B_1 (1x10) -> Rank 20
    # Part 2: A_2 (2x2) * B_2 (2x10) -> Rank 35 (5 blocks of Strassen 2x2x2)
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0 (first column of A, first row of B)
    # A_{i,0} * B_{0,k}
    for k in range(p):
        for i in range(n):
            # U index: i * m + j = i * 3 + 0
            u_idx = i * m + 0
            # V index: j * p + k = 0 * 10 + k
            v_idx = 0 * p + k
            # W index: k * n + i
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 2: j in {1, 2} (cols 1,2 of A, rows 1,2 of B)
    # A_2 is 2x2. B_2 is 2x10.
    # Split B_2 into 5 blocks of 2x2.
    # Strassen algorithm for each block.
    
    # A_2 indices in U:
    # A_{0,1} -> u_idx 0*3+1 = 1
    # A_{0,2} -> u_idx 0*3+2 = 2
    # A_{1,1} -> u_idx 1*3+1 = 4
    # A_{1,2} -> u_idx 1*3+2 = 5
    
    # B_2 indices in V:
    # B_{1, k} -> v_idx 1*10 + k = 10 + k
    # B_{2, k} -> v_idx 2*10 + k = 20 + k
    
    # C indices in W:
    # C_{i, k} -> w_idx k*2 + i
    
    # Strassen terms for A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4 + M7
    # C10 = M2 + M3 - M1 + M6
    # C11 = M1 - M2 + M5 - M6
    
    # Coefficients for W (C) for each M
    # M1: C00: 1, C01: 0, C10: -1, C11: 1
    # M2: C00: 0, C01: 0, C10: 1, C11: -1
    # M3: C00: 0, C01: 1, C10: 1, C11: 0
    # M4: C00: -1, C01: 1, C10: 0, C11: 0
    # M5: C00: 1, C01: 0, C10: 0, C11: 1
    # M6: C00: 0, C01: 0, C10: 1, C11: -1
    # M7: C00: -1, C01: 1, C10: 0, C11: 0
    
    # Linear forms for A (U)
    # M1: a+d -> U[1]=1, U[5]=1
    # M2: c+d -> U[4]=1, U[5]=1
    # M3: a   -> U[1]=1
    # M4: d   -> U[5]=1
    # M5: a+b -> U[1]=1, U[2]=1
    # M6: c-a -> U[4]=1, U[1]=-1
    # M7: b-d -> U[2]=1, U[5]=-1
    
    # Linear forms for B (V) for a block q (cols 2q, 2q+1)
    # e = B_{1, 2q} -> V[10+2q]
    # f = B_{1, 2q+1} -> V[10+2q+1]
    # g = B_{2, 2q} -> V[20+2q]
    # h = B_{2, 2q+1} -> V[20+2q+1]
    
    # M1: e+h -> V[10+2q]=1, V[20+2q+1]=1
    # M2: e   -> V[10+2q]=1
    # M3: f-h -> V[10+2q+1]=1, V[20+2q+1]=-1
    # M4: g-e -> V[20+2q]=1, V[10+2q]=-1
    # M5: h   -> V[20+2q+1]=1
    # M6: e+f -> V[10+2q]=1, V[10+2q+1]=1
    # M7: g+h -> V[20+2q]=1, V[20+2q+1]=1
    
    # Define the structure for the 7 terms
    # Each term is a dict of (u_idx, val), (v_idx, val), (w_idx, val)
    # But v and w indices depend on q.
    
    strassen_terms = [
        # M1
        {
            'u': [(1, 1), (5, 1)],
            'v': [(10, 1), (21, 1)], # e, h (offsets 0, 1 relative to 10, 20? No, explicit offsets)
            # e is 10+2q, h is 20+2q+1
            'w': [(0, 1), (1, -1), (2, 0), (3, 1)] # C00, C10, C01, C11 relative to 4q
        },
        # M2
        {
            'u': [(4, 1), (5, 1)],
            'v': [(10, 1)], # e
            'w': [(0, 0), (1, 1), (2, 0), (3, -1)]
        },
        # M3
        {
            'u': [(1, 1)],
            'v': [(11, 1), (21, -1)], # f, -h
            'w': [(0, 0), (1, 1), (2, 1), (3, 0)]
        },
        # M4
        {
            'u': [(5, 1)],
            'v': [(20, 1), (10, -1)], # g, -e
            'w': [(0, -1), (1, 0), (2, 1), (3, 0)]
        },
        # M5
        {
            'u': [(1, 1), (2, 1)],
            'v': [(21, 1)], # h
            'w': [(0, 1), (1, 0), (2, 0), (3, 1)]
        },
        # M6
        {
            'u': [(4, 1), (1, -1)],
            'v': [(10, 1), (11, 1)], # e, f
            'w': [(0, 0), (1, 1), (2, 0), (3, -1)]
        },
        # M7
        {
            'u': [(2, 1), (5, -1)],
            'v': [(20, 1), (21, 1)], # g, h
            'w': [(0, -1), (1, 0), (2, 1), (3, 0)]
        }
    ]
    
    # Note: The 'v' and 'w' offsets in the list above are relative to the base of the block q.
    # Base for V row 1 is 10, row 2 is 20.
    # Base for W is 4q.
    
    for q in range(5):
        for t in strassen_terms:
            # Update U
            for u_idx, val in t['u']:
                U[u_idx][col] += val
            
            # Update V
            for v_offset, val in t['v']:
                # v_offset is absolute row index in V?
                # My list above used absolute indices like 10, 11, 20, 21.
                # But these depend on q.
                # 10+2q, 10+2q+1, 20+2q, 20+2q+1.
                # I hardcoded 10, 11, 20, 21 in the list. I need to adjust.
                # Let's re-define the list with offsets relative to 10 and 20.
                pass
            
            # This hardcoded list approach is error-prone. Let's generate it dynamically.
            
            # Actually, I can just calculate the indices directly in the loop.
            
            # Let's restart the loop logic to be explicit and correct.
            pass

    # Corrected Loop for Part 2
    # Reset col? No, col is currently 20.
    
    for q in range(5):
        # Base indices for V
        v_e = 10 + 2*q
        v_f = 10 + 2*q + 1
        v_g = 20 + 2*q
        v_h = 20 + 2*q + 1
        
        # Base indices for W
        w_00 = 4*q
        w_10 = 4*q + 1
        w_01 = 4*q + 2
        w_11 = 4*q + 3
        
        # Term M1
        U[1][col] += 1
        U[5][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        W[w_00][col] += 1
        W[w_10][col] += -1
        W[w_11][col] += 1
        col += 1
        
        # Term M2
        U[4][col] += 1
        U[5][col] += 1
        V[v_e][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1
        
        # Term M3
        U[1][col] += 1
        V[v_f][col] += 1
        V[v_h][col] += -1
        W[w_10][col] += 1
        W[w_01][col] += 1
        col += 1
        
        # Term M4
        U[5][col] += 1
        V[v_g][col] += 1
        V[v_e][col] += -1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1
        
        # Term M5
        U[1][col] += 1
        U[2][col] += 1
        V[v_h][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # Term M6
        U[4][col] += 1
        U[1][col] += -1
        V[v_e][col] += 1
        V[v_f][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1
        
        # Term M7
        U[2][col] += 1
        U[5][col] += -1
        V[v_g][col] += 1
        V[v_h][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
