def construct():
    n, m, p = 2, 5, 6
    # Block 1: 2x2x6 schoolbook
    # A[:, 0:2], B[0:2, :]
    # Rank 24
    # Block 2a: 2x1x6 rank 1
    # A[:, 2:3], B[2:3, :]
    # Block 2b: 2x2x6 schoolbook
    # A[:, 3:5], B[3:5, :]
    # Rank 24
    
    total_rank = 24 + 1 + 24
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook 2x2x6
    def add_schoolbook_2x2x6(offset_a, offset_b):
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    U[(i * 2 + (j + offset_a)) * 1 + 0][col] = 1 # Wait, U index is i*m+j? No, U is (n*m, rank).
                    # U index: i * m + j. m=5.
                    # So U[i * 5 + (j + offset_a)][col] = 1
                    V[(j + offset_b) * 6 + k][col] = 1
                    W[k * 2 + i][col] = 1
                    col += 1
                    
    # Helper to add rank 1 2x1x6
    def add_rank1_2x1x6(offset_a, offset_b):
        nonlocal col
        for i in range(2):
            U[i * 5 + (offset_a)][col] = 1
        V[(offset_b) * 6 + 0][col] = 1 # Wait, V index is j*p+k. j=0, k=0?
        # For 2x1x6, j is 0. k is 0..5.
        # V index: j * 6 + k.
        # We need V[j, k] = 1 for all k?
        # T_{i, j, k} = 1.
        # Sum_r U_{ir} V_{jr} W_{kr} = 1.
        # If r is single column:
        # U_{i, r} = 1
        # V_{j, r} = 1
        # W_{k, r} = 1
        # But W index is k*n + i.
        # So W[k*2 + i, r] = 1.
        # V index is j*p + k.
        # Here j is fixed (0). k varies.
        # So V[0*6 + k, r] = 1 for all k?
        # No, V is (m*p, rank).
        # If we set V[0*6+k, r] = 1, then sum is sum_k 1*1*1 = 6?
        # No, the tensor entry is T_{i, j, k}.
        # The decomposition is sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}?
        # No, U is (n*m, rank). Index i*m+j.
        # V is (m*p, rank). Index j*p+k.
        # W is (p*n, rank). Index k*n+i.
        
        # For T_{i, j, k} = 1 (for all i, k, fixed j):
        # We need sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1.
        # If we use 1 term r:
        # U_{i*m+j, r} = 1
        # V_{j*p+k, r} = 1
        # W_{k*n+i, r} = 1
        # Then sum is 1.
        # So for fixed j, we need V_{j*p+k, r} = 1 for all k.
        # And U_{i*m+j, r} = 1 for all i.
        # And W_{k*n+i, r} = 1 for all k, i.
        
        # So:
        for i in range(2):
            U[i * 5 + offset_a][col] = 1
            for k in range(6):
                W[k * 2 + i][col] = 1
        for k in range(6):
            V[offset_b * 6 + k][col] = 1
        col += 1

    add_schoolbook_2x2x6(0, 0)
    add_rank1_2x1x6(2, 2)
    add_schoolbook_2x2x6(3, 3)
    
    return {"U": U, "V": V, "W": W}
