def construct():
    n, m, p = 4, 4, 8
    # M2 factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: 0=A11, 1=A12, 2=A21, 3=A22
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 1, 0, 0, 1, 0],  # B12
        [0, 0, 0, 1, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ]
    
    # M4 construction
    # 7 blocks, each 7 columns
    # Total 49 columns
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Strassen combos for M4 (using M2 blocks)
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We need to map global indices to local M2 indices and coefficients.
    # Global A index: a*4+b
    # Block A11: a in 0,1, b in 0,1. Local idx: a*2+b
    # Block A12: a in 0,1, b in 2,3. Local idx: a*2+(b-2)
    # Block A21: a in 2,3, b in 0,1. Local idx: (a-2)*2+b
    # Block A22: a in 2,3, b in 2,3. Local idx: (a-2)*2+(b-2)
    
    def get_local_idx_a(a, b, block):
        if block == 0: # A11
            return a*2+b
        elif block == 1: # A12
            return a*2+(b-2)
        elif block == 2: # A21
            return (a-2)*2+b
        elif block == 3: # A22
            return (a-2)*2+(b-2)
            
    def get_local_idx_b(b, c, block):
        if block == 0: # B11
            return b*2+c
        elif block == 1: # B12
            return b*2+(c-2)
        elif block == 2: # B21
            return (b-2)*2+c
        elif block == 3: # B22
            return (b-2)*2+(c-2)
            
    def get_local_idx_c(c, a, block):
        if block == 0: # C11
            return c*2+a
        elif block == 1: # C12
            return c*2+(a-2)
        elif block == 2: # C21
            return (c-2)*2+a
        elif block == 3: # C22
            return (c-2)*2+(a-2)
            
    # For each Strassen product k (0..6)
    for k in range(7):
        # Define X_k (A side) and Y_k (B side) as dict: block -> coeff
        # And C side: block -> coeff
        if k == 0: # P1
            X = {0: 1, 3: 1}
            Y = {0: 1, 3: 1}
            C = {0: 1, 3: 1} # P1 contributes to C11 and C22
            # C11 coeff 1, C22 coeff 1
            # But W2 handles the C block internal structure.
            # We need to add W2[k_local] * C_coeff to W4
        elif k == 1: # P2
            X = {2: 1, 3: 1}
            Y = {0: 1}
            C = {2: 1, 3: -1} # P2 to C21 (+1), C22 (-1)
        elif k == 2: # P3
            X = {0: 1}
            Y = {1: 1, 3: -1}
            C = {1: 1, 3: 1} # P3 to C12 (+1), C22 (+1)
        elif k == 3: # P4
            X = {3: 1}
            Y = {2: 1, 0: -1}
            C = {0: 1, 2: 1} # P4 to C11 (+1), C21 (+1)
        elif k == 4: # P5
            X = {0: 1, 1: 1}
            Y = {3: 1}
            C = {0: -1, 1: 1} # P5 to C11 (-1), C12 (+1)
        elif k == 5: # P6
            X = {2: 1, 0: -1}
            Y = {0: 1, 1: 1}
            C = {3: 1} # P6 to C22 (+1)
        elif k == 6: # P7
            X = {1: 1, 3: -1}
            Y = {2: 1, 3: 1}
            C = {0: 1} # P7 to C11 (+1)
            
        # Fill columns k*7 to k*7+6
        for r in range(7):
            col = k*7 + r
            
            # Fill U4
            for block, coeff in X.items():
                # Iterate over all global A indices in this block
                if block == 0:
                    rows = [0,1]; cols = [0,1]
                elif block == 1:
                    rows = [0,1]; cols = [2,3]
                elif block == 2:
                    rows = [2,3]; cols = [0,1]
                elif block == 3:
                    rows = [2,3]; cols = [2,3]
                
                for a in rows:
                    for b in cols:
                        global_i = a*4 + b
                        local_i = get_local_idx_a(a, b, block)
                        U4[global_i][col] += coeff * U2[local_i][r]
                        
            # Fill V4
            for block, coeff in Y.items():
                if block == 0:
                    rows = [0,1]; cols = [0,1]
                elif block == 1:
                    rows = [0,1]; cols = [2,3]
                elif block == 2:
                    rows = [2,3]; cols = [0,1]
                elif block == 3:
                    rows = [2,3]; cols = [2,3]
                
                for b in rows:
                    for c in cols:
                        global_j = b*4 + c
                        local_j = get_local_idx_b(b, c, block)
                        V4[global_j][col] += coeff * V2[local_j][r]
                        
            # Fill W4
            for block, coeff in C.items():
                if block == 0:
                    rows = [0,1]; cols = [0,1]
                elif block == 1:
                    rows = [0,1]; cols = [2,3]
                elif block == 2:
                    rows = [2,3]; cols = [0,1]
                elif block == 3:
                    rows = [2,3]; cols = [2,3]
                
                for c in rows:
                    for a in cols:
                        global_k = c*4 + a
                        local_k = get_local_idx_c(c, a, block)
                        W4[global_k][col] += coeff * W2[local_k][r]
                        
    # Now construct 4x4x8
    # U: 16x98
    # V: 32x98
    # W: 32x98
    
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    # Block 1 (cols 0..3 of C)
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            U[i][c+49] = U4[i][c]
            
    for j in range(16):
        for c in range(49):
            V[j][c] = V4[j][c]
            # V4[j] corresponds to B1 (cols 0..3)
            # V[j+16] corresponds to B2 (cols 4..7)
            
    for k in range(16):
        for c in range(49):
            W[k][c] = W4[k][c]
            # W[k] corresponds to C1 (cols 0..3)
            # W[k+16] corresponds to C2 (cols 4..7)
            
    for j in range(16):
        for c in range(49):
            V[j+16][c+49] = V4[j][c]
            
    for k in range(16):
        for c in range(49):
            W[k+16][c+49] = W4[k][c]
            
    return {"U": U, "V": V, "W": W}
