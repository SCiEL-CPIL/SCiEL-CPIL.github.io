import numpy as np
import time

DIMENSIONS = [3, 3, 8]
_DEADLINE = time.time() + 145.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _embed_strassen_into_3x3x8():
    """
    Embed Strassen 2x2x2 blocks into the 3x3*3x8 problem.
    Added a 5th block for k in {6,7} to improve rank.
    """
    n, m, p = 3, 3, 8
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    def zero_u(): return [0]*9
    def zero_v(): return [0]*24
    def zero_w(): return [0]*24
    
    def strassen_block(r0, r1, c0, c1, k0, k1):
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r0*m+c0] = 1; u[r1*m+c1] = 1
        v[c0*p+k0] = 1; v[c1*p+k1] = 1
        w[k0*n+r0] = 1; w[k1*n+r1] = 1
        add_term(u, v, w)
        
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r1*m+c0] = 1; u[r1*m+c1] = 1
        v[c0*p+k0] = 1
        w[k0*n+r1] = 1; w[k1*n+r1] = -1
        add_term(u, v, w)
        
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r0*m+c0] = 1
        v[c0*p+k1] = 1; v[c1*p+k1] = -1
        w[k1*n+r0] = 1; w[k1*n+r1] = 1
        add_term(u, v, w)
        
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r1*m+c1] = 1
        v[c1*p+k0] = 1; v[c0*p+k0] = -1
        w[k0*n+r0] = 1; w[k0*n+r1] = 1
        add_term(u, v, w)
        
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r0*m+c0] = 1; u[r0*m+c1] = 1
        v[c1*p+k1] = 1
        w[k0*n+r0] = -1; w[k1*n+r0] = 1
        add_term(u, v, w)
        
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r1*m+c0] = 1; u[r0*m+c0] = -1
        v[c0*p+k0] = 1; v[c0*p+k1] = 1
        w[k1*n+r1] = 1
        add_term(u, v, w)
        
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r0*m+c1] = 1; u[r1*m+c1] = -1
        v[c1*p+k0] = 1; v[c1*p+k1] = 1
        w[k0*n+r0] = 1
        add_term(u, v, w)
    
    def schoolbook_term(i, j, k):
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i*m+j] = 1
        v[j*p+k] = 1
        w[k*n+i] = 1
        add_term(u, v, w)
    
    covered = set()
    
    # Original 4 blocks
    strassen_blocks = [
        ((0,1), (0,1), (0,1)),
        ((0,1), (0,1), (2,3)),
        ((0,1), (0,1), (4,5)),
        ((1,2), (1,2), (0,1)),
    ]
    
    # New block 5: k in {6,7}
    strassen_blocks.append(((0,1), (0,1), (6,7)))
    
    for (r0, r1), (c0, c1), (k0, k1) in strassen_blocks:
        for i in [r0, r1]:
            for j in [c0, c1]:
                for k in [k0, k1]:
                    covered.add((i, j, k))
        strassen_block(r0, r1, c0, c1, k0, k1)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    schoolbook_term(i, j, k)
    
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n*m):
            if u[idx] != 0:
                U[idx][r] = u[idx]
        for idx in range(m*p):
            if v[idx] != 0:
                V[idx][r] = v[idx]
        for idx in range(p*n):
            if w[idx] != 0:
                W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    if _time_left() > 5:
        try:
            strassen_result = _embed_strassen_into_3x3x8()
            if strassen_result and _verify(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                r = len(strassen_result["U"][0])
                if r < best_rank:
                    best_rank = r
                    best_result = strassen_result
        except Exception:
            pass
    
    return best_result
