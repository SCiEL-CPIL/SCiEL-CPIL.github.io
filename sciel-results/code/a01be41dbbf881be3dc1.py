"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Strassen patterns for 2x2x2
    strassen_patterns = [
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    def compute_rank(n_splits, m_splits, p_splits):
        """Compute the rank for a given set of splits."""
        n_offsets = [0]
        for s in n_splits[:-1]:
            n_offsets.append(n_offsets[-1] + s)
        m_offsets = [0]
        for s in m_splits[:-1]:
            m_offsets.append(m_offsets[-1] + s)
        p_offsets = [0]
        for s in p_splits[:-1]:
            p_offsets.append(p_offsets[-1] + s)
        
        rank = 0
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]
                    
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        rank += 7  # Strassen
                    else:
                        rank += n_b * m_b * p_b  # Schoolbook
        
        return rank
    
    def build_decomposition(n_splits, m_splits, p_splits):
        """Build the full decomposition for given splits."""
        n_offsets = [0]
        for s in n_splits[:-1]:
            n_offsets.append(n_offsets[-1] + s)
        m_offsets = [0]
        for s in m_splits[:-1]:
            m_offsets.append(m_offsets[-1] + s)
        p_offsets = [0]
        for s in p_splits[:-1]:
            p_offsets.append(p_offsets[-1] + s)
        
        rank = compute_rank(n_splits, m_splits, p_splits)
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]
                    
                    i_off = n_offsets[i_b]
                    j_off = m_offsets[j_b]
                    k_off = p_offsets[k_b]
                    
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        for pattern in strassen_patterns:
                            # U
                            for li, lj, val in pattern['U']:
                                gi = i_off + li
                                gj = j_off + lj
                                U[gi * m + gj][col] = val
                            # V
                            for lj, lk, val in pattern['V']:
                                gj = j_off + lj
                                gk = k_off + lk
                                V[gj * p + gk][col] = val
                            # W
                            for lk, li, val in pattern['W']:
                                gk = k_off + lk
                                gi = i_off + li
                                W[gk * n + gi][col] = val
                            col += 1
                    else:
                        for li in range(n_b):
                            for lj in range(m_b):
                                for lk in range(p_b):
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1
        
        return {"U": U, "V": V, "W": W}, rank
    
    # Generate all possible split patterns for each dimension
    # For dimension d, find all ways to partition into parts of size 1 or 2
    def get_splits(d):
        """Get all possible splits of d into parts of 1 and 2."""
        splits = []
        def _gen(remaining, current):
            if remaining == 0:
                splits.append(current[:])
                return
            if remaining >= 1:
                current.append(1)
                _gen(remaining - 1, current)
                current.pop()
            if remaining >= 2:
                current.append(2)
                _gen(remaining - 2, current)
                current.pop()
        _gen(d, [])
        return splits
    
    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)
    
    # Sort by rank to try best first
    all_candidates = []
    for ns in n_split_options:
        for ms in m_split_options:
            for ps in p_split_options:
                r = compute_rank(ns, ms, ps)
                all_candidates.append((r, ns, ms, ps))
    
    all_candidates.sort(key=lambda x: x[0])
    
    # Try candidates in order of increasing rank
    best_result = None
    best_rank = float('inf')
    
    for r, ns, ms, ps in all_candidates:
        if time.time() > deadline:
            break
        if r >= best_rank:
            break  # No point trying higher rank
        
        try:
            result, actual_rank = build_decomposition(ns, ms, ps)
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: schoolbook
        best_result, _ = build_decomposition([n], [m], [p])
    
    return best_result
# EVOLVE-BLOCK-END
