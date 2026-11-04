def construct():
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Build 3x3 schoolbook
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = i*3 + j # Wait, schoolbook index mapping
                # Seed: U[i*m+j], V[j*p+k], W[k*n+i]
                # So c corresponds to (i,j,k)
                # In 3x3:
                c = i*3 + j # No, c is the column index in U3, V3, W3
                # The loop in seed:
                # for i, j, k:
                #    U[i*m+j][c] = 1
                #    V[j*p+k][c] = 1
                #    W[k*n+i][c] = 1
                # So c is just a counter from 0 to 26.
                pass
    
    # Correct 3x3 construction
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1
                
    # Strassen steps
    # Each step is a dict or tuple describing:
    # A_parts: list of (block_idx, coeff)
    # B_parts: list of (block_idx, coeff)
    # C_parts: list of (block_idx, coeff)
    # Block indices: 0:11, 1:12, 2:21, 3:22
    
    steps = [
        # M1
        { 'A': [(0,1), (3,1)], 'B': [(0,1), (3,1)], 'C': [(0,1), (3,1)] },
        # M2
        { 'A': [(2,1), (3,1)], 'B': [(0,1)], 'C': [(2,1)] }, # C21 = M2 + M4, so M2 goes to C21
        # M3
        { 'A': [(0,1)], 'B': [(1,1), (3,-1)], 'C': [(1,1)] }, # C12 = M3 + M5
        # M4
        { 'A': [(3,1)], 'B': [(2,1), (0,-1)], 'C': [(0,1), (2,1)] }, # C11 = M1+M4-M5+M7, C21 = M2+M4
        # M5
        { 'A': [(0,1), (1,1)], 'B': [(3,1)], 'C': [(0,-1), (1,1)] }, # C11 = ... -M5, C12 = M3+M5
        # M6
        { 'A': [(2,1), (0,-1)], 'B': [(0,1), (1,1)], 'C': [(3,1)] }, # C22 = M1-M2+M3+M6
        # M7
        { 'A': [(1,1), (3,-1)], 'B': [(2,1), (3,1)], 'C': [(0,1)] }  # C11 = ... +M7
    ]
    
    # Note: C parts need to be checked against Strassen formulas:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # My C mappings:
    # M1: C11 (+1), C22 (+1) -> Correct
    # M2: C21 (+1), C22 (-1) -> My step 2 says C: [(2,1)]. Missing C22.
    # M3: C12 (+1), C22 (+1) -> My step 3 says C: [(1,1)]. Missing C22.
    # M4: C11 (+1), C21 (+1) -> My step 4 says C: [(0,1), (2,1)]. Correct.
    # M5: C11 (-1), C12 (+1) -> My step 5 says C: [(0,-1), (1,1)]. Correct.
    # M6: C22 (+1) -> My step 6 says C: [(3,1)]. Correct.
    # M7: C11 (+1) -> My step 7 says C: [(0,1)]. Correct.
    
    # Fix M2: C21 (+1), C22 (-1)
    # Fix M3: C12 (+1), C22 (+1)
    
    steps[1]['C'] = [(2,1), (3,-1)]
    steps[2]['C'] = [(1,1), (3,1)]
    
    # Now populate global matrices
    for s_idx, step in enumerate(steps):
        for t in range(27):
            c_global = s_idx * 27 + t
            
            # Process U
            for i in range(3):
                for j in range(3):
                    u_val = U3[i*3+j][t]
                    if u_val != 0:
                        for block, coeff in step['A']:
                            # block 0: 11, 1: 12, 2: 21, 3: 22
                            # i, j are local
                            if block == 0: # 11
                                r = i
                                c = j
                            elif block == 1: # 12
                                r = i
                                c = j + 3
                            elif block == 2: # 21
                                r = i + 3
                                c = j
                            elif block == 3: # 22
                                r = i + 3
                                c = j + 3
                            U[r*6+c][c_global] += u_val * coeff
            
            # Process V
            for j in range(3):
                for k in range(3):
                    v_val = V3[j*3+k][t]
                    if v_val != 0:
                        for block, coeff in step['B']:
                            if block == 0: # 11
                                r = j
                                c = k
                            elif block == 1: # 12
                                r = j
                                c = k + 3
                            elif block == 2: # 21
                                r = j + 3
                                c = k
                            elif block == 3: # 22
                                r = j + 3
                                c = k + 3
                            V[r*6+c][c_global] += v_val * coeff
            
            # Process W
            for k in range(3):
                for i in range(3):
                    w_val = W3[k*3+i][t]
                    if w_val != 0:
                        for block, coeff in step['C']:
                            if block == 0: # 11
                                r = k
                                c = i
                            elif block == 1: # 12
                                r = k
                                c = i + 3
                            elif block == 2: # 21
                                r = k + 3
                                c = i
                            elif block == 3: # 22
                                r = k + 3
                                c = i + 3
                            W[r*6+c][c_global] += w_val * coeff

    return {"U": U, "V": V, "W": W}
