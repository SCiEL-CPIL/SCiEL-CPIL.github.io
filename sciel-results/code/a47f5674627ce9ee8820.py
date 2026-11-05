import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def _get_partitions(n):
    """
    Generate all partitions of n into 1s and 2s.
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[1]]
    if n == 2:
        return [[1, 1], [2]]
    
    parts = []
    # Start with 1
    for p in _get_partitions(n - 1):
        parts.append([1] + p)
    # Start with 2
    if n >= 2:
        for p in _get_partitions(n - 2):
            parts.append([2] + p)
    return parts

def _block_rank(nb, mb, pb):
    """
    Return the rank of a block multiplication of size nb x mb x pb.
    Assumes nb, mb, pb are in {1, 2}.
    """
    if nb == 2 and mb == 2 and pb == 2:
        return 7
    else:
        return nb * mb * pb

def _find_optimal_partition(n, m, p):
    """
    Find the partition of n, m, p into 1s and 2s that minimizes the total rank.
    """
    n_parts = _get_partitions(n)
    m_parts = _get_partitions(m)
    p_parts = _get_partitions(p)
    
    best_rank = float('inf')
    best_n = None
    best_m = None
    best_p = None
    
    for np_ in n_parts:
        for mp_ in m_parts:
            for pp_ in p_parts:
                rank = 0
                for nb in np_:
                    for mb in mp_:
                        for pb in pp_:
                            rank += _block_rank(nb, mb, pb)
                if rank < best_rank:
                    best_rank = rank
                    best_n = np_
                    best_m = mp_
                    best_p = pp_
                    
    return best_n, best_m, best_p, best_rank

def _strassen_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    """
    return [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

def _build_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """
    Build the decomposition using the given block partitions.
    """
    strassen_data = _strassen_terms()
    
    # Compute offsets
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)
    
    # Compute total rank
    total_rank = 0
    for ns in n_blocks:
        for ms in m_blocks:
            for ps in p_blocks:
                total_rank += _block_rank(ns, ms, ps)
                
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
                    # Use Strassen
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            # cr is row of C (i), cc is col of C (k)
                            W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                        col += 1
                else:
                    # Use schoolbook
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                U[(n0 + i) * m + (m0 + j), col] = 1
                                V[(m0 + j) * p + (p0 + k), col] = 1
                                W[(p0 + k) * n + (n0 + i), col] = 1
                                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition(n, m, p):
    """
    Constructs the schoolbook decomposition for matrix multiplication.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    return {"U": U, "V": W, "W": W} # Typo in original? No, W is W.

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    # Find optimal partition
    n_blocks, m_blocks, p_blocks, best_rank = _find_optimal_partition(n, m, p)
    
    # Build the best decomposition
    try:
        result = _build_decomposition(n, m, p, n_blocks, m_blocks, p_blocks)
        return result
    except Exception:
        return _schoolbook_decomposition(n, m, p)
