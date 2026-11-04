import numpy as np

def _embed(u, v, w, i_off, j_off, k_off, n, m, p, n_sub, m_sub, p_sub):
    """
    Embeds a local decomposition into global indices.
    u is local (n_sub*m_sub), v is local (m_sub*p_sub), w is local (p_sub*n_sub).
    i_off, j_off, k_off are offsets in the global i, j, k dimensions.
    """
    global_u = [0] * (n * m)
    global_v = [0] * (m * p)
    global_w = [0] * (p * n)
    
    for idx, val in enumerate(u):
        if val != 0:
            li, lj = divmod(idx, m_sub)
            gi, gj = i_off + li, j_off + lj
            global_u[gi * m + gj] = val
            
    for idx, val in enumerate(v):
        if val != 0:
            lj, lk = divmod(idx, p_sub)
            gj, gk = j_off + lj, k_off + lk
            global_v[gj * p + gk] = val
            
    for idx, val in enumerate(w):
        if val != 0:
            lk, li = divmod(idx, n_sub)
            gk, gi = k_off + lk, i_off + li
            global_w[gk * n + gi] = val
            
    return global_u, global_v, global_w

def _strassen_2x2():
    # Returns 7 terms for 2x2x2
    # u is 4x1, v is 4x1, w is 4x1 (flattened)
    # Indices: i,j for u; j,k for v; k,i for w
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    # A11: i=0,j=0; A22: i=1,j=1
    # B11: j=0,k=0; B22: j=1,k=1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # C11 += M1, C22 += M1
    # Wait, w maps to C.
    # C11 gets M1. C22 gets M1.
    # w index: k*n + i.
    # C11: k=0, i=0 -> idx 0.
    # C22: k=1, i=1 -> idx 3.
    # So w1 should have 1 at 0 and 3.
    w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))
    
    # M2 = (A21+A22)B11
    # A21: i=1,j=0; A22: i=1,j=1
    # B11: j=0,k=0
    u2 = [0, 1, 1, 0]
    v2 = [1, 0, 0, 0]
    # C21 += M2. C21: k=0, i=1 -> idx 1.
    w2 = [0, 1, 0, 0]
    terms.append((u2, v2, w2))
    
    # M3 = A11(B21-B22)
    # A11: i=0,j=0
    # B21: j=1,k=0; B22: j=1,k=1
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 1, -1] # j=1,k=0 is idx 2; j=1,k=1 is idx 3.
    # C12 += M3. C12: k=1, i=0 -> idx 2.
    w3 = [0, 0, 1, 0]
    terms.append((u3, v3, w3))
    
    # M4 = A22(B12-B11)
    # A22: i=1,j=1
    # B12: j=0,k=1; B11: j=0,k=0
    u4 = [0, 0, 0, 1]
    v4 = [-1, 1, 0, 0] # j=0,k=0 idx 0; j=0,k=1 idx 1.
    # C11 += M4. C11: k=0, i=0 -> idx 0.
    # C21 += M4. C21: k=0, i=1 -> idx 1.
    w4 = [1, 1, 0, 0]
    terms.append((u4, v4, w4))
    
    # M5 = (A11+A12)B22
    # A11: i=0,j=0; A12: i=0,j=1
    # B22: j=1,k=1
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    # C11 -= M5. C11: idx 0.
    # C12 += M5. C12: idx 2.
    w5 = [-1, 0, 1, 0]
    terms.append((u5, v5, w5))
    
    # M6 = (A21-A11)(B11+B12)
    # A21: i=1,j=0; A11: i=0,j=0
    # B11: j=0,k=0; B12: j=0,k=1
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    # C22 += M6. C22: idx 3.
    w6 = [0, 0, 0, 1]
    terms.append((u6, v6, w6))
    
    # M7 = (A12-A22)(B21+B22)
    # A12: i=0,j=1; A22: i=1,j=1
    # B21: j=1,k=0; B22: j=1,k=1
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    # C11 += M7. C11: idx 0.
    # C22 += M7. C22: idx 3.
    w7 = [1, 0, 0, 1]
    terms.append((u7, v7, w7))
    
    return terms

def _trivial_1x2x2():
    # 1x2x2
    terms = []
    for j in range(2):
        for k in range(2):
            u = [0, 0, 0, 0]
            u[j] = 1 # i=0, so idx j
            v = [0, 0, 0, 0]
            v[j*2 + k] = 1
            w = [0, 0]
            w[k*1 + 0] = 1
            terms.append((u, v, w))
    return terms

def _get_decomp(n, m, p):
    if n == 1 and m == 1 and p == 1:
        return [([1], [1], [1])]
    if n == 1 and m == 2 and p == 2:
        return _trivial_1x2x2()
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
        
    # Split strategy
    # If p > 2, split p into p1 + p2
    # If n > 2, split n
    # If m > 2, split m
    
    # Heuristic: split the largest dimension into powers of 2 if possible, or just 2 and rest.
    
    if p >= 2:
        p1 = 2
        p2 = p - 2
        if p2 == 0:
             # Handle 2x2x2 or 1x2x2
             if n == 2 and m == 2:
                 return _strassen_2x2()
             if n == 1 and m == 2:
                 return _trivial_1x2x2()
        d1 = _get_decomp(n, m, p1)
        d2 = _get_decomp(n, m, p2)
        
        res = []
        # Embed d1: k offset 0
        for u, v, w in d1:
            gu, gv, gw = _embed(u, v, w, 0, 0, 0, n, m, p, n, m, p1)
            res.append((gu, gv, gw))
        # Embed d2: k offset p1
        for u, v, w in d2:
            gu, gv, gw = _embed(u, v, w, 0, 0, p1, n, m, p, n, m, p2)
            res.append((gu, gv, gw))
        return res

    if n >= 2:
        n1 = 2
        n2 = n - 2
        d1 = _get_decomp(n1, m, p)
        d2 = _get_decomp(n2, m, p)
        
        res = []
        for u, v, w in d1:
            gu, gv, gw = _embed(u, v, w, 0, 0, 0, n, m, p, n1, m, p)
            res.append((gu, gv, gw))
        for u, v, w in d2:
            gu, gv, gw = _embed(u, v, w, n1, 0, 0, n, m, p, n2, m, p)
            res.append((gu, gv, gw))
        return res

    if m >= 2:
        m1 = 2
        m2 = m - 2
        d1 = _get_decomp(n, m1, p)
        d2 = _get_decomp(n, m2, p)
        
        res = []
        for u, v, w in d1:
            gu, gv, gw = _embed(u, v, w, 0, 0, 0, n, m, p, n, m1, p)
            res.append((gu, gv, gw))
        for u, v, w in d2:
            gu, gv, gw = _embed(u, v, w, 0, m1, 0, n, m, p, n, m2, p)
            res.append((gu, gv, gw))
        return res
    
    # Fallback trivial
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0]*(n*m); u[i*m+j] = 1
                v = [0]*(m*p); v[j*p+k] = 1
                w = [0]*(p*n); w[k*n+i] = 1
                terms.append((u, v, w))
    return terms

def construct():
    n, m, p = 3, 4, 6
    terms = _get_decomp(n, m, p)
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for idx, (u, v, w) in enumerate(terms):
        for i in range(n*m):
            if u[i] != 0:
                U[i][idx] = u[i]
        for i in range(m*p):
            if v[i] != 0:
                V[i][idx] = v[i]
        for i in range(p*n):
            if w[i] != 0:
                W[i][idx] = w[i]
                
    return {"U": U, "V": V, "W": W}
