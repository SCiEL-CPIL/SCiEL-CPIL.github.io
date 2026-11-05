def _build_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    strassen_data = _strassen_terms()
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)
    
    total_rank = _compute_rank(n_blocks, m_blocks, p_blocks)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                n0 = n_offsets[ni]
                m0 = m_offsets[mi]
                p0 = p_offsets[pi]
                ns = n_blocks[ni]
                ms = m_blocks[mi]
                ps = p_blocks[pi]
                
                if ns == 2 and ms == 2 and ps == 2:
                    # Strassen
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                        col += 1
                elif ns == 1:
                    # Rank ms
                    R = ms
                    for c in range(R):
                        for a in range(ms):
                            if a == c:
                                U[(n0) * m + (m0 + a), col] = 1
                        for b in range(ms):
                            if b == c:
                                for k in range(ps):
                                    V[(m0 + b) * p + (p0 + k), col] = 1
                        for k in range(ps):
                            W[(p0 + k) * n + (n0), col] = 1
                        col += 1
                elif ms == 1:
                    # Rank ps
                    R = ps
                    for c in range(R):
                        for i in range(ns):
                            U[(n0 + i) * m + (m0), col] = 1
                        for k in range(ps):
                            if k == c:
                                for b in range(ms): # b=0
                                    V[(m0 + b) * p + (p0 + k), col] = 1
                        for i in range(ns):
                            for k in range(ps):
                                if k == c:
                                    W[(p0 + k) * n + (n0 + i), col] = 1
                        col += 1
                elif ps == 1:
                    # Rank ms
                    R = ms
                    for c in range(R):
                        for a in range(ms):
                            if a == c:
                                for i in range(ns):
                                    U[(n0 + i) * m + (m0 + a), col] = 1
                        for b in range(ms):
                            if b == c:
                                V[(m0 + b) * p + (p0), col] = 1
                        for i in range(ns):
                            W[(p0) * n + (n0 + i), col] = 1
                        col += 1
                else:
                    # Schoolbook
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                U[(n0 + i) * m + (m0 + j), col] = 1
                                V[(m0 + j) * p + (p0 + k), col] = 1
                                W[(p0 + k) * n + (n0 + i), col] = 1
                                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
