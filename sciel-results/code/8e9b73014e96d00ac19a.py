import numpy as np
import time
import random

START_TIME = time.time()
TIME_LIMIT = 145.0

def remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _winograd_2x2x2():
    """Returns 7 rank-1 terms for 2x2x2 matmul."""
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [-1,0,1,0], [-1,1,1,0]),
        ([1,1,0,0], [0,0,1,0], [1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,-1,1]),
        ([0,1,0,-1], [0,1,0,1], [-1,1,1,0]),
    ]
    return terms

def _embed_2x2x2(terms, i_base, j_base, k_base, n, m, p):
    """Embed 2x2x2 Winograd terms into full-sized vectors."""
    full_terms = []
    for (u, v, w) in terms:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        for a in range(2):
            for b in range(2):
                idx = (i_base + a) * m + (j_base + b)
                ufull[idx] = u[a * 2 + b]
        for b in range(2):
            for c in range(2):
                idx = (j_base + b) * p + (k_base + c)
                vfull[idx] = v[b * 2 + c]
        for c in range(2):
            for a in range(2):
                idx = (k_base + c) * n + (i_base + a)
                wfull[idx] = w[c * 2 + a]
        full_terms.append((ufull, vfull, wfull))
    return full_terms

def _embed_2x1x2(i_base, j_idx, k_base, n, m, p):
    """Embed a 2x1x2 rank-1 term: C[i,k] = A[i,j]*B[j,k] for i in {i_base,i_base+1}, k in {k_base,k_base+1}."""
    ufull = [0] * (n * m)
    vfull = [0] * (m * p)
    wfull = [0] * (p * n)
    for a in range(2):
        ufull[(i_base + a) * m + j_idx] = 1
    for c in range(2):
        vfull[j_idx * p + (k_base + c)] = 1
    for c in range(2):
        for a in range(2):
            wfull[(k_base + c) * n + (i_base + a)] = 1
    return (ufull, vfull, wfull)

def _verify(U, V, W, rank):
    """Verify exact decomposition using random integer matrices."""
    n, m, p = 4, 5, 6
    for _ in range(5):
        if remaining() <= 0:
            return False
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        u_val = U[i * m + j][c]
                        if u_val == 0:
                            continue
                        acc += u_val * W[k * n + i][c] * V[j * p + k][c]
                if acc != C[i][k]:
                    return False
    return True

def _schoolbook():
    n, m, p = 4, 5, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}

def _optimized_decomposition():
    """Use 2x2x2 Winograd blocks for j<4 and 2x1x2 rank-1 terms for j=4."""
    n, m, p = 4, 5, 6
    terms_U = []
    terms_V = []
    terms_W = []
    
    winograd_222 = _winograd_2x2x2()
    
    # 2x2x2 Winograd blocks: i in {0,1},{2,3}; j in {0,1},{2,3}; k in {0,1},{2,3},{4,5}
    for ib in range(2):
        for jb in range(2):
            for kb in range(3):
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                embedded = _embed_2x2x2(winograd_222, i_base, j_base, k_base, n, m, p)
                for (u, v, w) in embedded:
                    terms_U.append(u)
                    terms_V.append(v)
                    terms_W.append(w)
    
    # 2x1x2 rank-1 terms for j=4: i pairs {0,1},{2,3}; k pairs {0,1},{2,3},{4,5}
    for ib in range(2):
        for kb in range(3):
            i_base = 2 * ib
            k_base = 2 * kb
            u, v, w = _embed_2x1x2(i_base, 4, k_base, n, m, p)
            terms_U.append(u)
            terms_V.append(v)
            terms_W.append(w)
    
    rank = len(terms_U)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
        for i in range(n * m):
            if u[i] != 0:
                U[i][c] = u[i]
        for j in range(m * p):
            if v[j] != 0:
                V[j][c] = v[j]
        for k in range(p * n):
            if w[k] != 0:
                W[k][c] = w[k]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _try_4x4_strassen_2():
    """
    Attempt to use a Strassen-like decomposition for the 4x4x2 part.
    This is a heuristic. We'll just use the 2x2x2 blocks for now if this fails.
    Actually, let's just stick to the 2x2x2 blocks which are proven.
    The 4x4x2 part is currently done with 2x2x2 blocks (6 blocks of 7 = 42 terms) + 4x1x6 (6 terms) = 48 terms.
    The 4x4x4 part is 4 blocks of 7 = 28 terms.
    Total 76 terms.
    """
    return None

def construct():
    n, m, p = 4, 5, 6
    
    # Try optimized decomposition first
    if remaining() > 10:
        try:
            result = _optimized_decomposition()
            if _verify(result["U"], result["V"], result["W"], result["rank"]):
                return {"U": result["U"], "V": result["V"], "W": result["W"]}
        except:
            pass
    
    # Fallback to schoolbook
    sb = _schoolbook()
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}
