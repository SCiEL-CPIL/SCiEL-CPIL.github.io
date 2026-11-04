import time

DIMENSIONS = [3, 4, 7]
_DEADLINE = time.time() + 140  # 10s safety margin

def _strassen_2x2x2():
    return [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {1:1, 3:1}),
        ({0:1}, {1:1, 3:-1}, {2:1, 3:-1}),
        ({3:1}, {2:1, 0:-1}, {1:1, 3:-1}),
        ({0:1, 1:1}, {3:1}, {2:1, 3:-1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {0:-1, 1:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:-1, 3:1}),
    ]

def _schoolbook(n, m, p):
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1}))
    return terms

def _decompose(n, m, p):
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    if p >= 2:
        p1, p2 = 2, p - 2
        if p2 == 0:
            p1, p2 = 1, 1
        terms1 = _decompose(n, m, p1)
        terms2 = _decompose(n, m, p2)
        remapped2 = []
        for U, V, W in terms2:
            V_new = {}
            for idx, val in V.items():
                j = idx // p2
                k = idx % p2
                V_new[j*p + (k + p1)] = val
            W_new = {}
            for idx, val in W.items():
                k_sub = idx // n
                i = idx % n
                W_new[(k_sub + p1)*n + i] = val
            remapped2.append((U, V_new, W_new))
        return terms1 + remapped2
    if m >= 2:
        m1, m2 = 2, m - 2
        if m2 == 0:
            m1, m2 = 1, 1
        terms1 = _decompose(n, m1, p)
        terms2 = _decompose(n, m2, p)
        remapped2 = []
        for U, V, W in terms2:
            U_new = {}
            for idx, val in U.items():
                i = idx // m2
                j = idx % m2
                U_new[i*m + (j + m1)] = val
            V_new = {}
            for idx, val in V.items():
                j = idx // p
                k = idx % p
                V_new[(j + m1)*p + k] = val
            remapped2.append((U_new, V_new, W))
        return terms1 + remapped2
    if n >= 2:
        n1, n2 = 2, n - 2
        if n2 == 0:
            n1, n2 = 1, 1
        terms1 = _decompose(n1, m, p)
        terms2 = _decompose(n2, m, p)
        remapped2 = []
        for U, V, W in terms2:
            U_new = {}
            for idx, val in U.items():
                i = idx // m
                j = idx % m
                U_new[(i + n1)*m + j] = val
            W_new = {}
            for idx, val in W.items():
                k = idx // n2
                i = idx % n2
                W_new[k*n + (i + n1)] = val
            remapped2.append((U_new, V, W_new))
        return terms1 + remapped2
    return _schoolbook(n, m, p)

def _verify(terms, n, m, p):
    """Verify that the decomposition correctly represents the multiplication tensor."""
    # Build the tensor: T[i*m+j, j*p+k, k*n+i] should be 1
    # Check: sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] == 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for U, V, W in terms:
                    u = U.get(i*m+j, 0)
                    v = V.get(j*p+k, 0)
                    w = W.get(k*n+i, 0)
                    total += u * v * w
                if total != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    terms = _decompose(n, m, p)
    
    # Verify
    if not _verify(terms, n, m, p):
        # Fallback to schoolbook
        terms = _schoolbook(n, m, p)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (U_dict, V_dict, W_dict) in enumerate(terms):
        for idx, val in U_dict.items():
            U[idx][r] = val
        for idx, val in V_dict.items():
            V[idx][r] = val
        for idx, val in W_dict.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
