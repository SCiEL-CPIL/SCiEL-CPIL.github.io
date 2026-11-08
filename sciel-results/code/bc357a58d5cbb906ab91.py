import time

def construct():
    # Constants
    N = 4
    M = 4
    P = 9
    
    # U: (16, R), V: (36, R), W: (36, R)
    # Indices: U[i*M + j, r], V[j*P + k, r], W[k*N + i, r]
    
    terms = []
    
    # Helper to add a term with given U, V, W vectors
    def _add(u, v, w):
        terms.append((u, v, w))
    
    # Strassen 2x2 definitions
    # Each term is (A_mask, B_mask, C_mask)
    # Masks are lists of (row, col, coeff)
    s2x2 = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    def _gen_4x4(b_off, c_off):
        """Generate 49 terms for 4x4 * 4x4 block."""
        block_terms = []
        for o_a, o_b, o_c in s2x2:
            for i_a, i_b, i_c in s2x2:
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36
                
                # Expand outer and inner
                for (ar, ac, ca) in o_a:
                    for (br, bc, cb) in i_a:
                        # A index: (2*ar + br, 2*ac + bc)
                        idx_u = (2*ar + br)*4 + (2*ac + bc)
                        u[idx_u] += ca * cb
                        
                for (ar, ac, ca) in o_b:
                    for (br, bc, cb) in i_b:
                        # B index: (2*ar + br, b_off + 2*ac + bc)
                        idx_v = (2*ar + br)*9 + (b_off + 2*ac + bc)
                        v[idx_v] += ca * cb
                        
                for (ar, ac, ca) in o_c:
                    for (br, bc, cb) in i_c:
                        # C index: (c_off + 2*ac + bc, 2*ar + br)
                        # W index: k*N + i -> (col)*4 + row
                        idx_w = (c_off + 2*ac + bc)*4 + (2*ar + br)
                        w[idx_w] += ca * cb
                        
                block_terms.append((u, v, w))
        return block_terms

    start = time.time()
    # Deadline guard
    deadline = start + 100
    
    if time.time() > deadline:
        # Fallback: Schoolbook 144
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    u = [0.0]*16
                    v = [0.0]*36
                    w = [0.0]*36
                    u[i*4+j] = 1.0
                    v[j*9+k] = 1.0
                    w[k*4+i] = 1.0
                    _add(u, v, w)
    else:
        # Block 1: cols 0-3 of B, cols 0-3 of C
        t1 = _gen_4x4(0, 0)
        # Block 2: cols 4-7 of B, cols 4-7 of C
        t2 = _gen_4x4(4, 4)
        
        for t in t1:
            _add(t[0], t[1], t[2])
        for t in t2:
            _add(t[0], t[1], t[2])
            
        # Block 3: col 8 of B, col 8 of C (4x1 mult)
        # Schoolbook for 4x4 * 4x1
        for i in range(4):
            for j in range(4):
                u = [0.0]*16
                v = [0.0]*36
                w = [0.0]*36
                u[i*4+j] = 1.0
                v[j*9+8] = 1.0
                w[8*4+i] = 1.0
                _add(u, v, w)

    # Filter out zero terms and combine identical terms
    def _is_zero(vec):
        return all(abs(x) < 1e-12 for x in vec)
    
    # Remove terms where any of U, V, W is entirely zero
    filtered_terms = []
    for t in terms:
        if not (_is_zero(t[0]) or _is_zero(t[1]) or _is_zero(t[2])):
            filtered_terms.append(t)
    
    # Try to combine terms with identical U vectors
    u_groups = {}
    for idx, t in enumerate(filtered_terms):
        u_key = tuple(t[0])
        if u_key not in u_groups:
            u_groups[u_key] = []
        u_groups[u_key].append(idx)
    
    # For each group with identical U, check if V and W can be combined
    final_terms = []
    used = set()
    for u_key, indices in u_groups.items():
        if len(indices) == 1:
            final_terms.append(filtered_terms[indices[0]])
            used.add(indices[0])
        else:
            # Check if all V vectors in the group are identical
            v_vectors = [tuple(filtered_terms[i][1]) for i in indices]
            w_vectors = [tuple(filtered_terms[i][2]) for i in indices]
            
            # If V vectors are all identical, we can combine by summing W
            if all(v == v_vectors[0] for v in v_vectors):
                # Sum all W vectors
                combined_w = [0.0] * 36
                for i in indices:
                    for k in range(36):
                        combined_w[k] += filtered_terms[i][2][k]
                final_terms.append((filtered_terms[indices[0]][0], filtered_terms[indices[0]][1], combined_w))
                used.update(indices)
            else:
                # Keep all terms
                for i in indices:
                    final_terms.append(filtered_terms[i])
                    used.add(i)
    
    # Also try combining by V vectors
    v_groups = {}
    for idx, t in enumerate(final_terms):
        v_key = tuple(t[1])
        if v_key not in v_groups:
            v_groups[v_key] = []
        v_groups[v_key].append(idx)
    
    final_terms2 = []
    used2 = set()
    for v_key, indices in v_groups.items():
        if len(indices) == 1:
            final_terms2.append(final_terms[indices[0]])
            used2.add(indices[0])
        else:
            # Check if all U vectors in the group are identical
            u_vectors = [tuple(final_terms[i][0]) for i in indices]
            w_vectors = [tuple(final_terms[i][2]) for i in indices]
            
            # If U vectors are all identical, we can combine by summing W
            if all(u == u_vectors[0] for u in u_vectors):
                # Sum all W vectors
                combined_w = [0.0] * 36
                for i in indices:
                    for k in range(36):
                        combined_w[k] += final_terms[i][2][k]
                final_terms2.append((final_terms[indices[0]][0], final_terms[indices[0]][1], combined_w))
                used2.update(indices)
            else:
                # Keep all terms
                for i in indices:
                    final_terms2.append(final_terms[i])
                    used2.add(i)
    
    terms = final_terms2
    
    R = len(terms)
    U = [[terms[r][0][i] for r in range(R)] for i in range(16)]
    V = [[terms[r][1][i] for r in range(R)] for i in range(36)]
    W = [[terms[r][2][i] for r in range(R)] for i in range(36)]
    
    return {"U": U, "V": V, "W": W}
