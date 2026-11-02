def _build_rank24():
    n, m, p = DIMENSIONS
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen terms defined with local indices
    # Local A: i*2+j (0..3)
    # Local B: j*2+k (0..3)
    # Local C: i*2+k (0..3) -> Note: W index is k*n+i, so local C index l=i*2+k maps to local W index k*2+i.
    # Let's stick to local C[i,k] indices l=i*2+k for the map, and convert to W index in loop.
    
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {2: 1, 0: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {1: 1, 2: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {2: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for b in range(3):
        for t_idx in range(7):
            col = b * 8 + t_idx
            u_map, v_map, w_map = strassen_terms[t_idx]
            
            for u_l, val in u_map.items():
                i_loc = u_l // 2
                j_loc = u_l % 2
                g_idx = i_loc * 3 + j_loc
                U[g_idx, col] = val
                
            for v_l, val in v_map.items():
                j_loc = v_l // 2
                k_loc = v_l % 2
                g_idx = j_loc * 6 + (2 * b + k_loc)
                V[g_idx, col] = val
                
            for w_l, val in w_map.items():
                i_loc = w_l // 2
                k_loc = w_l % 2
                g_idx = (2 * b + k_loc) * 2 + i_loc
                W[g_idx, col] = val
                
        # Outer term for j=2
        col = b * 8 + 7
        # U: i=0,1, j=2 -> global 2, 5
        U[2, col] = 1
        U[5, col] = 1
        # V: j=2, k=0,1 -> global 12+2b, 13+2b
        V[12 + 2 * b, col] = 1
        V[13 + 2 * b, col] = 1
        # W: k=0,1, i=0,1 -> global (2b)*2+0, (2b)*2+1, (2b+1)*2+0, (2b+1)*2+1
        W[(2 * b) * 2 + 0, col] = 1
        W[(2 * b) * 2 + 1, col] = 1
        W[(2 * b + 1) * 2 + 0, col] = 1
        W[(2 * b + 1) * 2 + 1, col] = 1

    return {"U": U, "V": V, "W": W}
