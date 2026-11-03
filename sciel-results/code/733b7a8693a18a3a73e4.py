import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Get terms for 4x4x4 Strassen
    # Each term is a dict: {'U': {idx: val}, 'V': {idx: val}, 'W': {idx: val}}
    terms_444 = get_strassen_terms_dict(4)
    
    rank = len(terms_444) * 2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Process first 4 columns
    for r, term in enumerate(terms_444):
        for u_idx, val in term['U'].items():
            U[u_idx][r] = val
        for v_idx, val in term['V'].items():
            # v_idx is local j*4+k
            j = v_idx // 4
            k = v_idx % 4
            V[j * p + k][r] = val
        for w_idx, val in term['W'].items():
            # w_idx is local k*4+i
            k = w_idx // 4
            i = w_idx % 4
            W[k * n + i][r] = val
            
    # Process last 4 columns
    offset = len(terms_444)
    for r, term in enumerate(terms_444):
        r_global = r + offset
        for u_idx, val in term['U'].items():
            U[u_idx][r_global] = val
        for v_idx, val in term['V'].items():
            j = v_idx // 4
            k = v_idx % 4
            V[j * p + (k + 4)][r_global] = val
        for w_idx, val in term['W'].items():
            k = w_idx // 4
            i = w_idx % 4
            W[(k + 4) * n + i][r_global] = val
            
    return {"U": U, "V": V, "W": W}

def get_strassen_terms_dict(N):
    if N == 1:
        return [{'U': {0: 1}, 'V': {0: 1}, 'W': {0: 1}}]
    
    M = N // 2
    terms_sub = get_strassen_terms_dict(M)
    
    terms = []
    
    def map_u(i, j, block):
        if block == 0: return i*M + j
        if block == 1: return i*M + (j+M)
        if block == 2: return (i+M)*M + j
        if block == 3: return (i+M)*M + (j+M)
        
    def map_v(j, k, block):
        if block == 0: return j*M + k
        if block == 1: return j*M + (k+M)
        if block == 2: return (j+M)*M + k
        if block == 3: return (j+M)*M + (k+M)
        
    def map_w(k, i, block):
        if block == 0: return k*M + i
        if block == 1: return k*M + (i+M)
        if block == 2: return (k+M)*M + i
        if block == 3: return (k+M)*M + (i+M)

    # M1 = (A11+A22)(B11+B22)
    for term in terms_sub:
        new_term = {'U': {}, 'V': {}, 'W': {}}
        for u_idx, val in term['U'].items():
            i, j = divmod(u_idx, M)
            # Add to A11 and A22
            idx1 = map_u(i, j, 0)
            idx2 = map_u(i, j, 3)
            new_term['U'][idx1] = new_term['U'].get(idx1, 0) + val
            new_term['U'][idx2] = new_term['U'].get(idx2, 0) + val
            
        for v_idx, val in term['V'].items():
            j, k = divmod(v_idx, M)
            idx1 = map_v(j, k, 0)
            idx2 = map_v(j, k, 3)
            new_term['V'][idx1] = new_term['V'].get(idx1, 0) + val
            new_term['V'][idx2] = new_term['V'].get(idx2, 0) + val
            
        for w_idx, val in term['W'].items():
            k, i = divmod(w_idx, M)
            idx1 = map_w(k, i, 0)
            idx2 = map_w(k, i, 3)
            new_term['W'][idx1] = new_term['W'].get(idx1, 0) + val
            new_term['W'][idx2] = new_term['W'].get(idx2, 0) + val
        terms.append(new_term)
        
    # M2 = (A21+A22)B11
    for term in terms_sub:
        new_term = {'U': {}, 'V': {}, 'W': {}}
        for u_idx, val in term['U'].items():
            i, j = divmod(u_idx, M)
            idx1 = map_u(i, j, 2)
            idx2 = map_u(i, j, 3)
            new_term['U'][idx1] = new_term['U'].get(idx1, 0) + val
            new_term['U'][idx2] = new_term['U'].get(idx2, 0) + val
        for v_idx, val in term['V'].items():
            j, k = divmod(v_idx, M)
            idx1 = map_v(j, k, 0)
            new_term['V'][idx1] = new_term['V'].get(idx1, 0) + val
        for w_idx, val in term['W'].items():
            k, i = divmod(w_idx, M)
            idx1 = map_w(k, i, 2)
            new_term['W'][idx1] = new_term['W'].get(idx1, 0) + val
        terms.append(new_term)
        
    # M3 = A11(B12-B22)
    for term in terms_sub:
        new_term = {'U': {}, 'V': {}, 'W': {}}
        for u_idx, val in term['U'].items():
            i, j = divmod(u_idx, M)
            idx1 = map_u(i, j, 0)
            new_term['U'][idx1] = new_term['U'].get(idx1, 0) + val
        for v_idx, val in term['V'].items():
            j, k = divmod(v_idx, M)
            idx1 = map_v(j, k, 1)
            idx2 = map_v(j, k, 3)
            new_term['V'][idx1] = new_term['V'].get(idx1, 0) + val
            new_term['V'][idx2] = new_term['V'].get(idx2, 0) - val
        for w_idx, val in term['W'].items():
            k, i = divmod(w_idx, M)
            idx1 = map_w(k, i, 1)
            new_term['W'][idx1] = new_term['W'].get(idx1, 0) + val
        terms.append(new_term)
        
    # M4 = A22(B21-B11)
    for term in terms_sub:
        new_term = {'U': {}, 'V': {}, 'W': {}}
        for u_idx, val in term['U'].items():
            i, j = divmod(u_idx, M)
            idx1 = map_u(i, j, 3)
            new_term['U'][idx1] = new_term['U'].get(idx1, 0) + val
        for v_idx, val in term['V'].items():
            j, k = divmod(v_idx, M)
            idx1 = map_v(j, k, 2)
            idx2 = map_v(j, k, 0)
            new_term['V'][idx1] = new_term['V'].get(idx1, 0) + val
            new_term['V'][idx2] = new_term['V'].get(idx2, 0) - val
        for w_idx, val in term['W'].items():
            k, i = divmod(w_idx, M)
            idx1 = map_w(k, i, 2)
            new_term['W'][idx1] = new_term['W'].get(idx1, 0) + val
        terms.append(new_term)
        
    # M5 = (A11+A12)B22
    for term in terms_sub:
        new_term = {'U': {}, 'V': {}, 'W': {}}
        for u_idx, val in term['U'].items():
            i, j = divmod(u_idx, M)
            idx1 = map_u(i, j, 0)
            idx2 = map_u(i, j, 1)
            new_term['U'][idx1] = new_term['U'].get(idx1, 0) + val
            new_term['U'][idx2] = new_term['U'].get(idx2, 0) + val
        for v_idx, val in term['V'].items():
            j, k = divmod(v_idx, M)
            idx1 = map_v(j, k, 3)
            new_term['V'][idx1] = new_term['V'].get(idx1, 0) + val
        for w_idx, val in term['W'].items():
            k, i = divmod(w_idx, M)
            idx1 = map_w(k, i, 1)
            new_term['W'][idx1] = new_term['W'].get(idx1, 0) + val
        terms.append(new_term)
        
    # M6 = (A21-A11)(B11+B12)
    for term in terms_sub:
        new_term = {'U': {}, 'V': {}, 'W': {}}
        for u_idx, val in term['U'].items():
            i, j = divmod(u_idx, M)
            idx1 = map_u(i, j, 2)
            idx2 = map_u(i, j, 0)
            new_term['U'][idx1] = new_term['U'].get(idx1, 0) + val
            new_term['U'][idx2] = new_term['U'].get(idx2, 0) - val
        for v_idx, val in term['V'].items():
            j, k = divmod(v_idx, M)
            idx1 = map_v(j, k, 0)
            idx2 = map_v(j, k, 1)
            new_term['V'][idx1] = new_term['V'].get(idx1, 0) + val
            new_term['V'][idx2] = new_term['V'].get(idx2, 0) + val
        for w_idx, val in term['W'].items():
            k, i = divmod(w_idx, M)
            idx1 = map_w(k, i, 0)
            new_term['W'][idx1] = new_term['W'].get(idx1, 0) + val
        terms.append(new_term)
        
    # M7 = (A12-A22)(B21+B22)
    for term in terms_sub:
        new_term = {'U': {}, 'V': {}, 'W': {}}
        for u_idx, val in term['U'].items():
            i, j = divmod(u_idx, M)
            idx1 = map_u(i, j, 1)
            idx2 = map_u(i, j, 3)
            new_term['U'][idx1] = new_term['U'].get(idx1, 0) + val
            new_term['U'][idx2] = new_term['U'].get(idx2, 0) - val
        for v_idx, val in term['V'].items():
            j, k = divmod(v_idx, M)
            idx1 = map_v(j, k, 2)
            idx2 = map_v(j, k, 3)
            new_term['V'][idx1] = new_term['V'].get(idx1, 0) + val
            new_term['V'][idx2] = new_term['V'].get(idx2, 0) + val
        for w_idx, val in term['W'].items():
            k, i = divmod(w_idx, M)
            idx1 = map_w(k, i, 3)
            new_term['W'][idx1] = new_term['W'].get(idx1, 0) + val
        terms.append(new_term)
        
    return terms

