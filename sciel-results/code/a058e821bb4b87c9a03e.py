DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition indices
    # U: (i,j) -> rank index
    # V: (j,k) -> rank index
    # W: (k,i) -> rank index
    # Values are +/- 1
    
    # r, u_idx, u_val, v_idx, v_val, w_idx, w_val
    # u_idx = i*2 + j
    # v_idx = j*2 + k
    # w_idx = k*2 + i
    
    strassen_terms = [
        # r=0
        (0, 0, 1, 0, 1, 0, 1), # U(0,0)=1, V(0,0)=1, W(0,0)=1
        # Wait, standard Strassen:
        # M1 = (A11+A22)(B11+B22) -> U: (0,0),(1,1). V: (0,0),(1,1). W: (0,0),(1,1) with coeffs.
        # Let's use the explicit 7 terms.
        
        # Term 0: (A11+A22)(B11+B22)
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: (0,0)=1, (1,1)=1 (M1)
        
        # Term 1: (A21+A22)B11
        # U: (1,0)=1, (1,1)=1
        # V: (0,0)=1
        # W: (0,1)=1, (1,1)=-1 (M1-M3-M5? No, standard indices)
        
        # Let's stick to the verified list from the current program, just restructured.
    ]
    
    # Verified Strassen Terms (i,j,k indices for U,V,W)
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    terms = [
        # r, [(u_idx, val), ...], [(v_idx, val), ...], [(w_idx, val), ...]
        (0, [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        (1, [(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        (2, [(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        (3, [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        (4, [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        (5, [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        (6, [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Define the 8 blocks
    # Block (bi, bj, bk) where bi in {0,1} (i offset), bj in {0,1,2,3} (j offset), bk in {0,1,2,3} (k offset)
    # Actually, the current code splits j into 0-1 and 2-3.
    # And k into 0-1, 2-3, 4-5, 6-7.
    # And i is always 0-1.
    
    blocks = [
        # (i_off, j_off, k_off, base_rank)
        (0, 0, 0, 0),
        (0, 0, 2, 7),
        (0, 0, 4, 14),
        (0, 0, 6, 21),
        (0, 2, 0, 28),
        (0, 2, 2, 35),
        (0, 2, 4, 42),
        (0, 2, 6, 49),
    ]
    
    for i_off, j_off, k_off, base_rank in blocks:
        for r, u_terms, v_terms, w_terms in terms:
            current_rank = base_rank + r
            
            for idx, val in u_terms:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_off
                global_j = local_j + j_off
                U[global_i * 4 + global_j][current_rank] += val
            
            for idx, val in v_terms:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_off
                global_k = local_k + k_off
                V[global_j * 8 + global_k][current_rank] += val
            
            for idx, val in w_terms:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_off
                global_i = local_i + i_off
                W[global_k * 2 + global_i][current_rank] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
