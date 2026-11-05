import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_coeffs():
    """Return list of (u, v, w) dicts for M222, rank 7.
    u: (i,j)->val, v: (j,k)->val, w: (k,i)->val"""
    terms = []
    # M1: (A00+A11)(B00+B11) -> C00 +=, C11 +=
    u = {(0,0):1,(1,1):1}; v = {(0,0):1,(1,1):1}; w = {(0,0):1,(1,1):1}
    terms.append((u,v,w))
    # M2: (A10+A11) B00 -> C10 +=, C11 -=
    u = {(1,0):1,(1,1):1}; v = {(0,0):1}; w = {(0,1):1,(1,1):-1}
    terms.append((u,v,w))
    # M3: A00 (B01-B11) -> C01 +=, C11 +=
    u = {(0,0):1}; v = {(0,1):1,(1,1):-1}; w = {(1,0):1,(1,1):1}
    terms.append((u,v,w))
    # M4: A11 (B10-B00) -> C00 +=, C10 +=
    u = {(1,1):1}; v = {(1,0):1,(0,0):-1}; w = {(0,0):1,(0,1):1}
    terms.append((u,v,w))
    # M5: (A00+A01) B11 -> C00 -=, C01 +=
    u = {(0,0):1,(0,1):1}; v = {(1,1):1}; w = {(0,0):-1,(1,0):1}
    terms.append((u,v,w))
    # M6: (A10-A00)(B00+B01) -> C10 +=
    u = {(1,0):1,(0,0):-1}; v = {(0,0):1,(0,1):1}; w = {(0,1):1}
    terms.append((u,v,w))
    # M7: (A01-A11)(B10+B11) -> C01 +=
    u = {(0,1):1,(1,1):-1}; v = {(1,0):1,(1,1):1}; w = {(1,0):1}
    terms.append((u,v,w))
    return terms

def _trivial_coeffs(ns, ms, ps):
    """Return list of (u,v,w) for trivial M_{ns,ms,ps}, rank ns*ms*ps."""
    terms = []
    for i in range(ns):
        for j in range(ms):
            for k in range(ps):
                u = {(i,j):1}; v = {(j,k):1}; w = {(k,i):1}
                terms.append((u,v,w))
    return terms

# Pre-compute coefficient tables for all needed block sizes
_COEFFS = {}
def _get_coeffs(ns, ms, ps):
    key = (ns, ms, ps)
    if key in _COEFFS:
        return _COEFFS[key]
    if ns == 2 and ms == 2 and ps == 2:
        _COEFFS[key] = _strassen_coeffs()
    else:
        _COEFFS[key] = _trivial_coeffs(ns, ms, ps)
    return _COEFFS[key]

def _compute_rank(n_blocks, m_blocks, p_blocks):
    rank = 0
    for ns, ms, ps in zip(n_blocks, m_blocks, p_blocks):
        if ns == 2 and ms == 2 and ps == 2:
            rank += 7
        else:
            rank += ns * ms * ps
    return rank

def _all_partitions(size, max_block=3):
    if size == 0:
        return [[]]
    results = []
    for b in range(1, min(max_block, size) + 1):
        for rest in _all_partitions(size - b, max_block):
            results.append([b] + rest)
    return results

def _build_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
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
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
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
                
                coeffs = _get_coeffs(ns, ms, ps)
                for u, v, w in coeffs:
                    for (i, j), val in u.items():
                        U[(n0 + i) * m + (m0 + j)][col] += val
                    for (j, k), val in v.items():
                        V[(m0 + j) * p + (p0 + k)][col] += val
                    for (k, i), val in w.items():
                        W[(p0 + k) * n + (n0 + i)][col] += val
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    n_partitions = _all_partitions(n)
    m_partitions = _all_partitions(m)
    p_partitions = _all_partitions(p)
    
    best_rank = n * m * p
    best_partition = (n_partitions[0], m_partitions[0], p_partitions[0])
    
    for np_ in n_partitions:
        for mp_ in m_partitions:
            for pp_ in p_partitions:
                r = _compute_rank(np_, mp_, pp_)
                if r < best_rank:
                    best_rank = r
                    best_partition = (np_, mp_, pp_)
    
    if time.time() > deadline - 20:
        return _schoolbook_decomposition(n, m, p)
    
    try:
        result = _build_decomposition(n, m, p, *best_partition)
        if _validate_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
        else:
            return _schoolbook_decomposition(n, m, p)
    except Exception:
        return _schoolbook_decomposition(n, m, p)
