<<<<<<< SEARCH
    # --- Combine ---
    # U is (16, 57). Both parts use full A.
    U = np.hstack([U_44, U_42])
    
    # V is (24, 57).
    # B is (4, 6). Flattened B has 24 elements.
    # B_1 is first 4 cols (16 elements). B_2 is last 2 cols (8 elements).
    # V_44 acts on B_1 (rows 0-15 of V).
    # V_42 acts on B_2 (rows 16-23 of V).
    V = np.zeros((24, 57), dtype=float)
    V[:16, :49] = V_44
    V[16:, 49:] = V_42
    
    # W is (24, 57).
    # C is (4, 6). Flattened C has 24 elements.
    # Convention W[k*n + i, r]. n=4.
    # C_1 is cols 0-3. k in 0..3. Indices k*4+i are 0..15.
    # C_2 is cols 4-5. k in 4..5. Indices k*4+i are 16..23.
    # W_44 acts on C_1 (rows 0-15 of W).
    # W_42 acts on C_2 (rows 16-23 of W).
    W = np.zeros((24, 57), dtype=float)
    W[:16, :49] = W_44
    W[16:, 49:] = W_42
=======
    # --- Combine ---
    # U is (16, 57). Both parts use full A.
    U = np.hstack([U_44, U_42])
    
    # V is (24, 57).
    # Assuming B is flattened Row-Major (consistent with A's i*4+k).
    # B is 4x6.
    # B_1 (cols 0-3) local index l = 4*k + j. Global index g = 6*k + j.
    # B_2 (cols 4-5) local index l = 2*k + j. Global index g = 6*k + 4 + j.
    
    V = np.zeros((24, 57), dtype=float)
    
    # Map V_44 rows to global B indices
    for l in range(16):
        k = l // 4
        j = l % 4
        g = 6 * k + j
        V[g, :49] = V_44[l, :]
        
    # Map V_42 rows to global B indices
    for l in range(8):
        k = l // 2
        j = l % 2
        g = 6 * k + 4 + j
        V[g, 49:] = V_42[l, :]
    
    # W is (24, 57).
    # C is (4, 6). Flattened C has 24 elements.
    # Convention W[k*n + i, r]. n=4. Column-Major.
    # C_1 is cols 0-3. k in 0..3. Indices k*4+i are 0..15.
    # C_2 is cols 4-5. k in 4..5. Indices k*4+i are 16..23.
    # W_44 acts on C_1 (rows 0-15 of W).
    # W_42 acts on C_2 (rows 16-23 of W).
    W = np.zeros((24, 57), dtype=float)
    W[:16, :49] = W_44
    W[16:, 49:] = W_42
>>>>>>> REPLACE
