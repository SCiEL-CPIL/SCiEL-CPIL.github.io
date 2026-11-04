import numpy as np
import time
from itertools import product

_start_time = time.time()

def construct():
    n, m, p = 3, 5, 6
    
    # Cache for min_rank
    min_rank_cache = {}
    
    def get_partitions(total):
        if total == 0:
            return [[]]
        results = []
        for first in range(1, total + 1):
            for rest in get_partitions(total - first):
                results.append([first] + rest)
        return results

    def min_rank(n_s, m_s, p_s):
        key = (n_s, m_s, p_s)
        if key in min_rank_cache:
            return min_rank_cache[key]
        
        if n_s == 1:
            rank = m_s * p_s
        elif m_s == 1:
            rank = n_s * p_s
        elif p_s == 1:
            rank = n_s * m_s
        elif n_s == 2 and m_s == 2 and p_s == 2:
            rank = 7
        else:
            best = float('inf')
            # Try all partitions
            for pn in get_partitions(n_s):
                for pm in get_partitions(m_s):
                    for pp in get_partitions(p_s):
                        # Check if partition is valid (sums to n_s, etc)
                        # get_partitions ensures this
                        curr_rank = 0
                        valid = True
                        for ns, ms, ps in zip(pn, pm, pp):
                            curr_rank += min_rank(ns, ms, ps)
                        if curr_rank < best:
                            best = curr_rank
            rank = best
        
        min_rank_cache[key] = rank
        return rank

    # Find best partition for 3, 5, 6
    best_rank = float('inf')
    best_partition = None
    
    for pn in get_partitions(n):
        for pm in get_partitions(m):
            for pp in get_partitions(p):
                rank = 0
                for ns, ms, ps in zip(pn, pm, pp):
                    rank += min_rank(ns, ms, ps)
                if rank < best_rank:
                    best_rank = rank
                    best_partition = (pn, pm, pp)
    
    # Construct matrices
    U = np.zeros((n*m, best_rank))
    V = np.zeros((m*p, best_rank))
    W = np.zeros((p*n, best_rank))
    
    col_offset = 0
    n_offsets = [0]
    for sz in best_partition[0][:-1]:
        n_offsets.append(n_offsets[-1] + sz)
    m_offsets = [0]
    for sz in best_partition[1][:-1]:
        m_offsets.append(m_offsets[-1] + sz)
    p_offsets = [0]
    for sz in best_partition[2][:-1]:
        p_offsets.append(p_offsets[-1] + sz)
        
    for ni, ns in enumerate(best_partition[0]):
        for mi, ms in enumerate(best_partition[1]):
            for pi, ps in enumerate(best_partition[2]):
                # Get block decomposition
                block_U, block_V, block_W, block_rank = build_block(ns, ms, ps)
                
                # Map to global
                for r in range(block_rank):
                    for i in range(ns):
                        for j in range(ms):
                            U[n_offsets[ni] + i, m_offsets[mi] + j] # Wrong indexing
                            # U row index: (global_i * m + global_j)
                            # global_i = n_offsets[ni] + i
                            # global_j = m_offsets[mi] + j
                            U[(n_offsets[ni] + i) * m + (m_offsets[mi] + j), col_offset + r] = block_U[i * ms + j, r]
                            
                    for j in range(ms):
                        for k in range(ps):
                            V[(m_offsets[mi] + j) * p + (p_offsets[pi] + k), col_offset + r] = block_V[j * ps + k, r]
                            
                    for k in range(ps):
                        for i in range(ns):
                            W[(p_offsets[pi] + k) * n + (n_offsets[ni] + i), col_offset + r] = block_W[k * ns + i, r]
                            
                col_offset += block_rank
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def build_block(n_s, m_s, p_s):
    # Recursive build
    if n_s == 1:
        U = np.zeros((n_s*m_s, m_s*p_s))
        V = np.zeros((m_s*p_s, m_s*p_s))
        W = np.zeros((p_s*n_s, m_s*p_s))
        # Rank 1? No, rank mp.
        # Schoolbook
        rank = m_s * p_s
        U = np.zeros((n_s*m_s, rank))
        V = np.zeros((m_s*p_s, rank))
        W = np.zeros((p_s*n_s, rank))
        for j in range(m_s):
            for k in range(p_s):
                r = j * p_s + k
                U[0, r] = 1 # n_s=1, i=0
                V[j*p_s+k, r] = 1
                W[k*n_s, r] = 1
        return U, V, W, rank
        
    # ... implement schoolbook and strassen
