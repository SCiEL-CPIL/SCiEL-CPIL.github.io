"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 5 * 7 (Strassen) + 1 * 4 (Schoolbook) = 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition vectors
    # u, v, w are lists of 7 tuples
    # u: (i,j) -> index i*m+j
    # v: (j,k) -> index j*p+k
    # w: (k,i) -> index k*n+i
    
    # Indices for u: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for v (local): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for w (local): 00, 10, 01, 11 -> 0, 1, 2, 3 (k*n+i)
    
    strassen_u = [
        (1, 0, 0, 1), # M1
        (0, 0, 1, 1), # M2
        (1, 0, 0, 0), # M3
        (0, 0, 0, 1), # M4
        (1, 1, 0, 0), # M5
        (0, 0, 1, 1), # M6
        (1, 0, -1, 0) # M7
    ]
    
    strassen_v_local = [
        (1, 1, 1, 1), # M1
        (1, 0, 0, 0), # M2
        (-1, 1, 0, 0), # M3
        (0, 0, -1, 1), # M4
        (0, 0, 0, 1), # M5
        (0, 0, 1, -1), # M6
        (1, 0, 1, 0) # M7
    ]
    
    strassen_w_local = [
        (1, 0, 0, 1), # M1
        (0, 1, 0, 0), # M2
        (0, 1, 0, 0), # M3
        (-1, 0, 1, 0), # M4
        (0, 0, 1, 0), # M5
        (0, 0, 0, 1), # M6
        (1, 0, 0, 0) # M7
    ]
    
    col = 0
    # 5 Strassen blocks
    for block in range(5):
        k_offset = block * 2
        for t in range(7):
            # Update U
            for idx, val in enumerate(strassen_u[t]):
                if val != 0:
                    i, j = divmod(idx, m)
                    U[i*m + j][col] += val
                    
            # Update V
            for idx, val in enumerate(strassen_v_local[t]):
                if val != 0:
                    j, k_local = divmod(idx, 2) # local k is 0,1
                    k_global = k_offset + k_local
                    V[j*p + k_global][col] += val
                    
            # Update W
            for idx, val in enumerate(strassen_w_local[t]):
                if val != 0:
                    k_local, i = divmod(idx, 2) # local k is 0,1
                    k_global = k_offset + k_local
                    W[k_global*n + i][col] += val
                    
            col += 1
            
    # 1 Schoolbook block (2x2x1)
    k_offset = 10
    school_u = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1)
    ]
    school_v_local = [
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1)
    ]
    school_w_local = [
        (1, 0),
        (1, 0),
        (0, 1),
        (0, 1)
    ]
    
    for t in range(4):
        for idx, val in enumerate(school_u[t]):
            if val != 0:
                i, j = divmod(idx, m)
                U[i*m + j][col] += val
                
        for idx, val in enumerate(school_v_local[t]):
            if val != 0:
                j, k_local = divmod(idx, 1)
                k_global = k_offset + k_local
                V[j*p + k_global][col] += val
                
        for idx, val in enumerate(school_w_local[t]):
            if val != 0:
                k_local, i = divmod(idx, 2) # w has 2 elements (k=0, i=0,1)
                # Wait, school_w_local indices:
                # (1,0) -> k=0, i=0
                # (0,1) -> k=0, i=1
                # So idx 0 is (0,0), idx 1 is (0,1)
                k_global = k_offset
                W[k_global*n + i][col] += val
                # Correction: divmod(idx, 2) is wrong for size 2.
                # idx 0 -> k=0, i=0
                # idx 1 -> k=0, i=1
                # So i = idx, k_local = 0.
                
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
