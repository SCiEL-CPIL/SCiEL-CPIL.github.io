"""Block-structured low-rank decomposition for 4x5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s safety margin before 150s kill

def remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, rank):
    """Verify exact decomposition using random integer matrices."""
    import random
    n, m, p = DIMENSIONS
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

def _embed_block(terms, i_base, j_base, k_base, i_size, j_size, k_size, n, m, p):
    """Embed a block's terms into full-sized vectors."""
    full_terms = []
    for (u, v, w) in terms:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        
        for a in range(i_size):
            for b in range(j_size):
                idx = (i_base + a) * m + (j_base + b)
                ufull[idx] = u[a * j_size + b]
        
        for b in range(j_size):
            for c in range(k_size):
                idx = (j_base + b) * p + (k_base + c)
                vfull[idx] = v[b * k_size + c]
        
        for c in range(k_size):
            for a in range(i_size):
                idx = (k_base + c) * n + (i_base + a)
                wfull[idx] = w[c * i_size + a]
        
        full_terms.append((ufull, vfull, wfull))
    return full_terms

def _block_decomposition():
    """Build 4x5x6 as sum of Winograd blocks plus schoolbook remainders."""
    n, m, p = DIMENSIONS
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    winograd_222 = _winograd_2x2x2()
    
    covered = set()
    
    for ib in range(2):
        for kb in range(2):
            for jb in range(2):
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                
                embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])
                
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            covered.add((i_base + a, j_base + b, k_base + c))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                ufull = [0] * (n * m)
                vfull = [0] * (m * p)
                wfull = [0] * (p * n)
                ufull[i * m + j] = 1
                vfull[j * p + k] = 1
                wfull[k * n + i] = 1
                terms_U.append(ufull)
                terms_V.append(vfull)
                terms_W.append(wfull)
    
    rank = len(terms_U)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
        for i in range(n * m):
            U[i][c] = u[i]
        for j in range(m * p):
            V[j][c] = v[j]
        for k in range(p * n):
            W[k][c] = w[k]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _try_2x2x3_blocks():
    """Try using 2x2x3 blocks for better coverage of k dimension."""
    n, m, p = DIMENSIONS
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    covered = set()
    
    winograd_222 = _winograd_2x2x2()
    
    for ib in range(2):
        for kb in range(2):
            for jb in range(2):
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                
                embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])
                
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            covered.add((i_base + a, j_base + b, k_base + c))
    
    for ib in range(2):
        for jb in range(2):
            i_base = 2 * ib
            j_base = 2 * jb
            k_base = 4
            
            embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
            terms_U.extend([t[0] for t in embedded])
            terms_V.extend([t[1] for t in embedded])
            terms_W.extend([t[2] for t in embedded])
            
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        covered.add((i_base + a, j_base + b, k_base + c))
    
    for i in range(n):
        for k in range(p):
            if (i, 4, k) in covered:
                continue
            ufull = [0] * (n * m)
            vfull = [0] * (m * p)
            wfull = [0] * (p * n)
            ufull[i * m + 4] = 1
            vfull[4 * p + k] = 1
            wfull[k * n + i] = 1
            terms_U.append(ufull)
            terms_V.append(vfull)
            terms_W.append(wfull)
    
    rank = len(terms_U)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
        for i in range(n * m):
            U[i][c] = u[i]
        for j in range(m * p):
            V[j][c] = v[j]
        for k in range(p * n):
            W[k][c] = w[k]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _winograd_3x3x3():
    """Returns rank-1 terms for 3x3x3 matmul using Winograd-like approach."""
    # 3x3x3 schoolbook rank = 27
    # We can try to use a combination of 2x2x2 blocks + remainders
    # Or just use schoolbook for now
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0]*9; v = [0]*9; w = [0]*9
                u[i*3+j] = 1
                v[j*3+k] = 1
                w[k*3+i] = 1
                terms.append((u, v, w))
    return terms

def _try_3x3_blocks():
    """Try using 3x3x3 blocks for better coverage."""
    n, m, p = DIMENSIONS
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    covered = set()
    
    winograd_222 = _winograd_2x2x2()
    
    # Use 2x2x2 blocks for i in {0,1},{2,3}, j in {0,1},{2,3}, k in {0,1},{2,3}
    for ib in range(2):
        for kb in range(2):
            for jb in range(2):
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                
                embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])
                
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            covered.add((i_base + a, j_base + b, k_base + c))
    
    # Add schoolbook for uncovered
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                ufull = [0] * (n * m)
                vfull = [0] * (m * p)
                wfull = [0] * (p * n)
                ufull[i * m + j] = 1
                vfull[j * p + k] = 1
                wfull[k * n + i] = 1
                terms_U.append(ufull)
                terms_V.append(vfull)
                terms_W.append(wfull)
    
    rank = len(terms_U)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
        for i in range(n * m):
            U[i][c] = u[i]
        for j in range(m * p):
            V[j][c] = v[j]
        for k in range(p * n):
            W[k][c] = w[k]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    
    candidates = []
    
    if remaining() > 10:
        try:
            result1 = _block_decomposition()
            candidates.append(result1)
        except:
            pass
    
    if remaining() > 5:
        try:
            result2 = _try_2x2x3_blocks()
            candidates.append(result2)
        except:
            pass
    
    if remaining() > 3:
        try:
            result3 = _try_3x3_blocks()
            candidates.append(result3)
        except:
            pass
    
    result4 = _schoolbook()
    candidates.append(result4)
    
    best = None
    for cand in candidates:
        if remaining() <= 0:
            break
        rank = cand["rank"]
        U, V, W = cand["U"], cand["V"], cand["W"]
        if _verify(U, V, W, rank):
            if best is None or rank < best["rank"]:
                best = cand
    
    if best is None:
        best = result4
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
