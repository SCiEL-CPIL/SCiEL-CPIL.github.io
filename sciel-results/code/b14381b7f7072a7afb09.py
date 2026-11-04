def construct():
    n, m, p = DIMENSIONS
    rank = 58
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        for idx, coeff in u_entries:
            U[idx][col] += coeff
        for idx, coeff in v_entries:
            V[idx][col] += coeff
        for idx, coeff in w_entries:
            W[idx][col] += coeff
        col += 1
    
    # 4x4 Strassen: 49 terms
    # Define the 2x2 Strassen template
    # Each term is (u_pattern, v_pattern, w_pattern)
    # where patterns are dicts from (i,j) to coeff
    
    # 2x2 Strassen:
    # T0: u={(0,0):1,(1,1):1}, v={(0,0):1,(1,1):1}, w={(0,0):1,(1,1):1}
    # T1: u={(1,0):1,(1,1):1}, v={(0,0):1}, w={(1,0):1,(1,1):1}
    # T2: u={(0,0):1}, v={(1,0):1,(1,1):-1}, w={(0,1):1,(1,1):1}
    # T3: u={(1,1):1}, v={(0,1):1,(0,0):-1}, w={(0,0):1,(1,1):1}
    # T4: u={(0,0):1,(0,1):1}, v={(1,1):1}, w={(0,0):1,(0,1):1}
    # T5: u={(1,0):1,(1,1):-1}, v={(0,0):1,(1,0):1}, w={(1,1):1}
    # T6: u={(0,1):1,(0,0):-1}, v={(0,1):1,(1,1):1}, w={(0,0):1}
    
    strassen_2x2 = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):1}),
        ({(0,0):1}, {(1,0):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(0,1):1,(0,0):-1}, {(0,0):1,(1,1):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):1,(0,1):1}),
        ({(1,0):1,(1,1):-1}, {(0,0):1,(1,0):1}, {(1,1):1}),
        ({(0,1):1,(0,0):-1}, {(0,1):1,(1,1):1}, {(0,0):1}),
    ]
    
    # 4x4 Strassen: 7 block-level terms
    # S0: X=A00+A11, Y=B00+B11, contributes to C00(+1), C11(+1)
    # S1: X=A10+A11, Y=B00, contributes to C10(+1), C11(+1)
    # S2: X=A00, Y=B10-B11, contributes to C01(+1), C11(+1)
    # S3: X=A11, Y=B01-B00, contributes to C00(+1), C11(+1)
    # S4: X=A00+A01, Y=B11, contributes to C00(+1), C01(+1)
    # S5: X=A10-A11, Y=B00+B10, contributes to C11(+1)
    # S6: X=A01-A00, Y=B01+B11, contributes to C00(+1)
    
    # For each S_i, X and Y are 2x2 matrices of 2x2 blocks.
    # X[xi][xj] is a sum of 2x2 A-blocks with coefficients.
    # Y[yi][yj] is a sum of 2x2 B-blocks with coefficients.
    
    # Define block patterns: for each S_i, X_blocks[xi][xj] = list of (block_row, block_col, coeff)
    # block (br, bc) corresponds to A[br*2:br*2+2, bc*2:bc*2+2]
    
    block_patterns = [
        # S0: X = A00+A11, Y = B00+B11
        ({(0,0): [(0,0,1),(1,1,1)], (0,1): [], (1,0): [], (1,1): []},
         {(0,0): [(0,0,1),(1,1,1)], (0,1): [], (1,0): [], (1,1): []}),
        # S1: X = A10+A11, Y = B00
        ({(0,0): [], (0,1): [], (1,0): [(1,0,1),(1,1,1)], (1,1): []},
         {(0,0): [(0,0,1)], (0,1): [], (1,0): [], (1,1): []}),
        # S2: X = A00, Y = B10-B11
        ({(0,0): [(0,0,1)], (0,1): [], (1,0): [], (1,1): []},
         {(0,0): [], (0,1): [], (1,0): [(1,0,1)], (1,1): [(1,1,-1)]}),
        # S3: X = A11, Y = B01-B00
        ({(0,0): [], (0,1): [], (1,0): [], (1,1): [(1,1,1)]},
         {(0,0): [(0,0,-1)], (0,1): [(0,1,1)], (1,0): [], (1,1): []}),
        # S4: X = A00+A01, Y = B11
        ({(0,0): [(0,0,1)], (0,1): [(0,1,1)], (1,0): [], (1,1): []},
         {(0,0): [], (0,1): [], (1,0): [], (1,1): [(1,1,1)]}),
        # S5: X = A10-A11, Y = B00+B10
        ({(0,0): [], (0,1): [], (1,0): [(1,0,1),(1,1,-1)], (1,1): []},
         {(0,0): [(0,0,1)], (0,1): [], (1,0): [(1,0,1)], (1,1): []}),
        # S6: X = A01-A00, Y = B01+B11
        ({(0,0): [(0,0,-1)], (0,1): [(0,1,1)], (1,0): [], (1,1): []},
         {(0,0): [], (0,1): [(0,1,1)], (1,0): [], (1,1): [(1,1,1)]}),
    ]
    
    # W contribution for each S_i:
    # C00 = S0 + S3 - S4 + S6
    # C01 = S2 + S4
    # C10 = S1 + S3
    # C11 = S0 - S1 + S2 + S5
    
    # For S_i, the W contribution is a 2x2 pattern:
    # S0: C00(+1), C11(+1) -> w_pat = {(0,0):1, (1,1):1}
    # S1: C10(+1), C11(+1) -> w_pat = {(1,0):1, (1,1):1}
    # S2: C01(+1), C11(+1) -> w_pat = {(0,1):1, (1,1):1}
    # S3: C00(+1), C11(+1) -> w_pat = {(0,0):1, (1,1):1}
    # S4: C00(-1), C01(+1) -> w_pat = {(0,0):-1, (0,1):1}
    # S5: C11(+1) -> w_pat = {(1,1):1}
    # S6: C00(+1) -> w_pat = {(0,0):1}
    
    s_w_patterns = [
        {(0,0):1, (1,1):1},  # S0
        {(1,0):1, (1,1):1},  # S1
        {(0,1):1, (1,1):1},  # S2
        {(0,0):1, (1,1):1},  # S3
        {(0,0):-1, (0,1):1}, # S4
        {(1,1):1},            # S5
        {(0,0):1},            # S6
    ]
    
    # For each S_i and each 2x2 Strassen term T_j, generate one rank-1 term
    for si, (X_blocks, Y_blocks) in enumerate(block_patterns):
        w_pat = s_w_patterns[si]
        for ti, (u_pat, v_pat, t_w_pat) in enumerate(strassen_2x2):
            # u_entries: for each (xi,xj) in u_pat, add the corresponding A-block entries
            u_entries = []
            for (xi, xj), coeff in u_pat.items():
                for (br, bc, bc_coeff) in X_blocks.get((xi, xj), []):
                    for li in range(2):
                        for lj in range(2):
                            gi = br * 2 + li
                            gj = bc * 2 + lj
                            u_entries.append((gi * m + gj, coeff * bc_coeff))
            
            v_entries = []
            for (yi, yj), coeff in v_pat.items():
                for (br, bc, bc_coeff) in Y_blocks.get((yi, yj), []):
                    for li in range(2):
                        for lj in range(2):
                            gi = br * 2 + li
                            gj = bc * 2 + lj
                            v_entries.append((gi * p + gj, coeff * bc_coeff))
            
            w_entries = []
            for (wi, wj), coeff in t_w_pat.items():
                # (wi, wj) is the local (i,j) in the 2x2 result of S_i
                # The global C index is: block_row = wi, block_col = wj
                # So global (i,j) = (wi*2 + local_i, wj*2 + local_j)
                # But wait, the 2x2 result of S_i is a 2x2 matrix, and each entry
                # is a 2x2 block of C.
                # So for local (wi, wj), the global C block is (wi, wj),
                # which covers C[wi*2:wi*2+2, wj*2:wj*2+2].
                # The W index is k*n + i, so for C[gi, gj], W index = gj*n + gi.
                for li in range(2):
                    for lj in range(2):
                        gi = wi * 2 + li
                        gj = wj * 2 + lj
                        # Multiply by the S_i W pattern coefficient
                        s_coeff = w_pat.get((wi, wj), 0)
                        if s_coeff != 0:
                            w_entries.append((gj * n + gi, coeff * s_coeff))
            
            add_term(u_entries, v_entries, w_entries)
    
    # Border terms: 9 terms
    # C[i,4] for i=0..3
    for i in range(4):
        u_entries = [(i * m + j, 1) for j in range(5)]
        v_entries = [(j * p + 4, 1) for j in range(5)]
        w_entries = [(4 * n + i, 1)]
        add_term(u_entries, v_entries, w_entries)
    
    # C[4,k] for k=0..3
    for k in range(4):
        u_entries = [(4 * m + j, 1) for j in range(5)]
        v_entries = [(j * p + k, 1) for j in range(5)]
        w_entries = [(k * n + 4, 1)]
        add_term(u_entries, v_entries, w_entries)
    
    # C[4,4]
    u_entries = [(4 * m + j, 1) for j in range(5)]
    v_entries = [(j * p + 4, 1) for j in range(5)]
    w_entries = [(4 * n + 4, 1)]
    add_term(u_entries, v_entries, w_entries)
    
    return {"U": U, "V": V, "W": W}
