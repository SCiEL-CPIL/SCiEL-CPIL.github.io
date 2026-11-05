import time
import numpy as np
from itertools import product

def construct():
    start = time.time()
    deadline = start + 100  # Safety margin
    
    n, m, p = 3, 5, 7
    
    # Generate all partitions of a number into parts of size 1 or 2
    def all_partitions(size):
        if size == 0:
            return [[]]
        results = []
        for rest in all_partitions(size - 1):
            results.append([1] + rest)
        if size >= 2:
            for rest in all_partitions(size - 2):
                results.append([2] + rest)
        return results
    
    def compute_rank(np_, mp_, pp_):
        rank = 0
        for ns, ms, ps in zip(np_, mp_, pp_):
            # Wait, this is wrong. We need to pair up blocks.
            # Actually, the partition of n, m, p are independent.
            # For each combination of (ni_block, mi_block, pi_block), we compute the rank.
            pass
        # Correct: iterate over all combinations
        rank = 0
        for ni in range(len(np_)):
            for mi in range(len(mp_)):
                for pi in range(len(pp_)):
                    ns = np_[ni]
                    ms = mp_[mi]
                    ps = pp_[pi]
                    if ns == 2 and ms == 2 and ps == 2:
                        rank += 7
                    else:
                        rank += ns * ms * ps
        return rank
    
    # Strassen 2x2x2 decomposition
    def get_strassen():
        # Returns U, V, W for 2x2x2 with Strassen (rank 7)
        # Indexing: U[i*2+a, c], V[b*2+k, c], W[k*2+i, c]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00 -> C10+=1, C11-=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11) -> C01+=1, C11+=1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00) -> C00+=1, C10+=1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11) -> C00+=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    sU, sV, sW = get_strassen()
    
    # Build decomposition for given partitions
    def build_decomp(np_, mp_, pp_):
        # Compute offsets
        n_off = [0]
        for s in np_:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in mp_:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in pp_:
            p_off.append(p_off[-1] + s)
        
        total_rank = 0
        for ni in range(len(np_)):
            for mi in range(len(mp_)):
                for pi in range(len(pp_)):
                    ns, ms, ps = np_[ni], mp_[mi], pp_[pi]
                    if ns == 2 and ms == 2 and ps == 2:
                        total_rank += 7
                    else:
                        total_rank += ns * ms * ps
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col = 0
        for ni in range(len(np_)):
            for mi in range(len(mp_)):
                for pi in range(len(pp_)):
                    n0, m0, p0 = n_off[ni], m_off[mi], p_off[pi]
                    ns, ms, ps = np_[ni], mp_[mi], pp_[pi]
                    
                    if ns == 2 and ms == 2 and ps == 2:
                        # Use Strassen
                        for c in range(7):
                            for i in range(2):
                                for a in range(2):
                                    U[(n0 + i) * m + (m0 + a), col] += sU[i*2+a, c]
                            for b in range(2):
                                for k in range(2):
                                    V[(m0 + b) * p + (p0 + k), col] += sV[b*2+k, c]
                            for i in range(2):
                                for k in range(2):
                                    W[(p0 + k) * n + (n0 + i), col] += sW[k*2+i, c]
                            col += 1
                    else:
                        # Schoolbook
                        for i in range(ns):
                            for a in range(ms):
                                for k in range(ps):
                                    U[(n0 + i) * m + (m0 + a), col] = 1
                                    V[(m0 + a) * p + (p0 + k), col] = 1
                                    W[(p0 + k) * n + (n0 + i), col] = 1
                                    col += 1
        
        if col != total_rank:
            raise ValueError(f"Column mismatch: {col} != {total_rank}")
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank
    
    # Generate partitions
    n_parts = all_partitions(n)
    m_parts = all_partitions(m)
    p_parts = all_partitions(p)
    
    # Find best partition
    best_rank = n * m * p
    best_np, best_mp, best_pp = n_parts[0], m_parts[0], p_parts[0]
    
    for np_ in n_parts:
        for mp_ in m_parts:
            for pp_ in p_parts:
                r = compute_rank(np_, mp_, pp_)
                if r < best_rank:
                    best_rank = r
                    best_np, best_mp, best_pp = np_, mp_, pp_
                
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    # Build the best decomposition
    try:
        result, rank = build_decomp(best_np, best_mp, best_pp)
        return result
    except Exception:
        pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                U[i * m + a, c] = 1
                V[a * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
