"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Strassen patterns for 2x2x2 (7 rank-1 terms)
    # U, V, W are lists of (index1, index2, value)
    # Indices are local to the 2x2x2 block
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
    
    def get_splits(d):
        """Generate all compositions of d into parts of size 1, 2, or 3."""
        results = []
        def helper(remaining, current):
            if remaining == 0:
                results.append(current[:])
                return
            for part in [1, 2, 3]:
                if remaining >= part:
                    current.append(part)
                    helper(remaining - part, current)
                    current.pop()
        helper(d, [])
        return results
    
    def compute_rank(n_splits, m_splits, p_splits):
        """Compute the rank for given splits, matching actual build logic."""
        rank = 0
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    if nb == 2 and mb == 2 and pb == 2:
                        rank += 7
                    else:
                        rank += nb * mb * pb
        return rank

    def build_decomposition(n_splits, m_splits, p_splits):
        """Build the full decomposition for given splits."""
        n_off = [0]
        for s in n_splits[:-1]:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in m_splits[:-1]:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in p_splits[:-1]:
            p_off.append(p_off[-1] + s)
        
        U_entries = []
        V_entries = []
        W_entries = []
        rank = 0
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb, mb, pb = n_splits[i_b], m_splits[j_b], p_splits[k_b]
                    i_off = n_off[i_b]
                    j_off = m_off[j_b]
                    k_off = p_off[k_b]
                    
                    if nb == 2 and mb == 2 and pb == 2:
                        for pattern in strassen_patterns:
                            col = rank
                            for li, lj, val in pattern['U']:
                                U_entries.append(((i_off + li) * m + (j_off + lj), col, val))
                            for lj, lk, val in pattern['V']:
                                V_entries.append(((j_off + lj) * p + (k_off + lk), col, val))
                            for lk, li, val in pattern['W']:
                                W_entries.append(((k_off + lk) * n + (i_off + li), col, val))
                            rank += 1
                    else:
                        # For non-2x2x2 blocks, use schoolbook
                        for li in range(nb):
                            for lj in range(mb):
                                for lk in range(pb):
                                    col = rank
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U_entries.append((gi * m + gj, col, 1))
                                    V_entries.append((gj * p + gk, col, 1))
                                    W_entries.append((gk * n + gi, col, 1))
                                    rank += 1
        
        # Build matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for idx, col, val in U_entries:
            U[idx][col] = val
        for idx, col, val in V_entries:
            V[idx][col] = val
        for idx, col, val in W_entries:
            W[idx][col] = val
        
        return {"U": U, "V": V, "W": W}, rank
    
    # Generate all possible splits for each dimension (now allowing 3s)
    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)
    
    # Sort splits by estimated rank (prefer 2s, then 3s, then 1s)
    def split_score(splits):
        """Lower score = better."""
        score = 0
        for s in splits:
            if s == 2:
                score += 0
            elif s == 3:
                score += 2
            else:  # s == 1
                score += 3
        return score
    
    n_split_options.sort(key=split_score)
    m_split_options.sort(key=split_score)
    p_split_options.sort(key=split_score)
    
    # Find the best combination of splits (minimum rank)
    best_rank = float('inf')
    best_result = None
    best_splits = None
    checked = 0
    
    for ns in n_split_options:
        if time.time() > deadline:
            break
        for ms in m_split_options:
            if time.time() > deadline:
                break
            for ps in p_split_options:
                if time.time() > deadline:
                    break
                # Quick rank estimate to skip obviously bad combinations
                est_rank = compute_rank(ns, ms, ps)
                if est_rank >= best_rank:
                    continue
                result, r = build_decomposition(ns, ms, ps)
                checked += 1
                if r < best_rank:
                    best_rank = r
                    best_result = result
                    best_splits = (ns, ms, ps)
    
    # Verify the decomposition is correct
    def verify_decomposition(result):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        U = np.array(result['U'])
        V = np.array(result['V'])
        W = np.array(result['W'])
        
        rank = U.shape[1]
        if U.shape[0] != n * m:
            return False
        if V.shape[0] != m * p:
            return False
        if W.shape[0] != p * n:
            return False
        if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    if best_result is not None:
        if not verify_decomposition(best_result):
            best_result = None
    
    if best_result is None:
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
        
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
