def construct():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0]*rank for _ in range(n*m)]  # 4 x 21
    V = [[0]*rank for _ in range(m*p)]  # 12 x 21
    W = [[0]*rank for _ in range(p*n)]  # 12 x 21
    
    # Strassen A-combinations (indexed 0-6)
    # A1 = a00+a11, A2 = a10+a11, A3 = a00, A4 = a11, A5 = a00+a01, A6 = a10-a00, A7 = a01-a11
    A_combos = [
        # (i, j, coeff) for A[i,j]
        [(0,0,1),(1,1,1)],       # r=0: a00+a11
        [(1,0,1),(1,1,1)],       # r=1: a10+a11
        [(0,0,1)],                # r=2: a00
        [(1,1,1)],                # r=3: a11
        [(0,0,1),(0,1,1)],       # r=4: a00+a01
        [(1,0,1),(0,0,-1)],      # r=5: a10-a00
        [(0,1,1),(1,1,-1)],      # r=6: a01-a11
    ]
    
    # For each block b (0,1,2) and Strassen product r (0-6)
    col = 0
    for b in range(3):
        for r in range(7):
            # Set U
            for (i,j,c) in A_combos[r]:
                U[i*m+j][col] = c
            
            # Set V: B-combinations for block b
            # Block b covers columns 2b and 2b+1
            c0 = 2*b
            c1 = 2*b+1
            B_combos = [
                [(0,c0,1),(1,c1,1)],       # r=0: b00+b11
                [(0,c0,1)],                 # r=1: b00
                [(1,c0,1),(1,c1,-1)],      # r=2: b10-b11
                [(0,c1,1),(0,c0,-1)],      # r=3: b01-b00
                [(1,c1,1)],                 # r=4: b11
                [(0,c0,1),(1,c0,1)],       # r=5: b00+b10
                [(0,c1,1),(1,c1,1)],       # r=6: b01+b11
            ]
            for (j,k,c) in B_combos[r]:
                V[j*p+k][col] = c
            
            # Set W: output contributions
            # C00_b = M1+M4-M5+M7 -> output (0, c0)
            # C01_b = M3+M5 -> output (0, c1)
            # C10_b = M2+M4 -> output (1, c0)
            # C11_b = M1-M2+M3+M6 -> output (1, c1)
            W_contribs = [
                # (k, i, coeff) for W[k*n+i]
                [(c0,0,1),(c1,0,-1),(c1,1,-1),(c0,1,1)],   # r=0: +C00, -C01? No wait...
            ]
            # Let me redo this more carefully
            # For Strassen product r, what does it contribute to?
            # M_r contributes to specific C entries with specific signs
            # 
            # M0 (M1): C00 += 1, C11 += 1
            # M1 (M2): C10 += 1, C11 -= 1
            # M2 (M3): C01 += 1, C11 += 1
            # M3 (M4): C00 += 1, C10 += 1
            # M4 (M5): C00 -= 1, C01 += 1
            # M5 (M6): C11 += 1
            # M6 (M7): C00 += 1
            
            W_map = [
                [(c0,0,1),(c1,1,1)],       # r=0: M1 -> C00 +1, C11 +1
                [(c0,1,1),(c1,1,-1)],      # r=1: M2 -> C10 +1, C11 -1
                [(c1,0,1),(c1,1,1)],       # r=2: M3 -> C01 +1, C11 +1
                [(c0,0,1),(c0,1,1)],       # r=3: M4 -> C00 +1, C10 +1
                [(c0,0,-1),(c1,0,1)],      # r=4: M5 -> C00 -1, C01 +1
                [(c1,1,1)],                 # r=5: M6 -> C11 +1
                [(c0,0,1)],                 # r=6: M7 -> C00 +1
            ]
            for (k,i,c) in W_map[r]:
                W[k*n+i][col] = c
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
