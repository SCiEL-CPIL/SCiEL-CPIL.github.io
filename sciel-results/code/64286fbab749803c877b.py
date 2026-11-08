def construct():
    # ...
    n, m, p = 5, 5, 5
    # Get Strassen factors for 4x4x4
    U_4, V_4, W_4 = get_strassen_4x4()
    # U_4 shape (16, 49), V_4 shape (16, 49), W_4 shape (16, 49)
    # Indices in U_4 are i*4+j for i,j in 0..3
    
    # Initialize full factors
    rank_total = 49 + 61
    U = np.zeros((25, rank_total))
    V = np.zeros((25, rank_total))
    W = np.zeros((25, rank_total))
    
    # Copy Strassen part
    U[:, :49] = np.zeros((25, 49)) # Need to map indices
    # Actually, U_4 has rows 0..15 corresponding to i,j in 0..3.
    # In U_full, rows 0..15 correspond to i,j in 0..3.
    # So we can just assign U[0:16, 0:49] = U_4
    U[0:16, 0:49] = U_4
    V[0:16, 0:49] = V_4
    W[0:16, 0:49] = W_4
    
    # Add schoolbook for remaining
    col = 49
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i < 4 and j < 4 and k < 4:
                    continue
                U[i*5+j, col] = 1.0
                V[j*5+k, col] = 1.0
                W[k*5+i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
