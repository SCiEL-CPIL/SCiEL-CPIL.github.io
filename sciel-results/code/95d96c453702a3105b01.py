"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen (2x2)
    # l, x_inds, x_sigs, y_inds, y_sigs
    inner_strassen = [
        (0, [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        (1, [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
        (2, [(0,0)], [1], [(0,1), (1,1)], [1, -1]),
        (3, [(1,1)], [1], [(1,0), (0,0)], [1, -1]),
        (4, [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        (5, [(1,0), (1,1)], [1, -1], [(0,0), (0,1)], [1, 1]),
        (6, [(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1]),
    ]
    
    # Reconstruction 2x2
    # z_rc, m_inds, m_sigs
    recon_2x2 = [
        ((0,0), [0, 3, 1, 2], [1, 1, -1, 1]),
        ((0,1), [2, 4], [1, 1]),
        ((1,0), [5, 1], [1, 1]),
        ((1,1), [0, 5, 3, 6], [1, -1, 1, 1]),
    ]
    
    # Outer Strassen (4x4 blocks)
    # k, a_blocks, a_sigs, b_blocks, b_sigs
    outer_strassen = [
        (0, [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        (1, [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
        (2, [(0,0)], [1], [(0,1), (1,1)], [1, -1]),
        (3, [(1,1)], [1], [(1,0), (0,0)], [1, -1]),
        (4, [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        (5, [(1,0), (1,1)], [1, -1], [(0,0), (0,1)], [1, 1]),
        (6, [(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1]),
    ]
    
    # Reconstruction 4x4
    # c_block_rc, m_inds, m_sigs
    recon_4x4 = [
        ((0,0), [0, 3, 1, 2], [1, 1, -1, 1]),
        ((0,1), [2, 4], [1, 1]),
        ((1,0), [5, 1], [1, 1]),
        ((1,1), [0, 5, 3, 6], [1, -1, 1, 1]),
    ]
    
    for k in range(7):
        _, a_blocks, a_sigs, b_blocks, b_sigs = outer_strassen[k]
        
        # X_k, Y_k are 2x2 matrices of dicts
        X_k = [[{} for _ in range(2)] for _ in range(2)]
        Y_k = [[{} for _ in range(2)] for _ in range(2)]
        
        for r in range(2):
            for c in range(2):
                for idx, (bi, bj) in enumerate(a_blocks):
                    sign = a_sigs[idx]
                    a_idx = (bi * 2 + r) * 4 + (bj * 2 + c)
                    X_k[r][c][a_idx] = X_k[r][c].get(a_idx, 0) + sign
                    
                for idx, (bi, bj) in enumerate(b_blocks):
                    sign = b_sigs[idx]
                    v_idx = (bi * 2 + r) * 4 + (bj * 2 + c)
                    Y_k[r][c][v_idx] = Y_k[r][c].get(v_idx, 0) + sign

        for l in range(7):
            col = k * 7 + l
            _, x_inds, x_sigs, y_inds, y_sigs = inner_strassen[l]
            
            L_A = {}
            L_B = {}
            
            for idx, (xr, xc) in enumerate(x_inds):
                sign = x_sigs[idx]
                for a_idx, coeff in X_k[xr][xc].items():
                    val = coeff * sign
                    L_A[a_idx] = L_A.get(a_idx, 0) + val
                    
            for idx, (yr, yc) in enumerate(y_inds):
                sign = y_sigs[idx]
                for v_idx, coeff in Y_k[yr][yc].items():
                    val = coeff * sign
                    L_B[v_idx] = L_B.get(v_idx, 0) + val
            
            for u_idx, val in L_A.items():
                if val != 0:
                    U[u_idx][col] += val
            for v_idx, val in L_B.items():
                if val != 0:
                    V[v_idx][col] += val
            
            # Determine W contributions
            m_contribs = {}
            for z_rc, m_inds, m_sigs in recon_2x2:
                if l in m_inds:
                    idx = m_inds.index(l)
                    m_contribs[z_rc] = m_sigs[idx]
                    
            for z_rc, coeff_m in m_contribs.items():
                z_r, z_c = z_rc
                for c_block_rc, m_inds, m_sigs in recon_4x4:
                    if k in m_inds:
                        idx = m_inds.index(k)
                        coeff_c = m_sigs[idx]
                        cb_r, cb_c = c_block_rc
                        i = cb_r * 2 + z_r
                        k_prime = cb_c * 2 + z_c
                        w_idx = k_prime * 4 + i
                        W[w_idx][col] += coeff_m * coeff_c

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
