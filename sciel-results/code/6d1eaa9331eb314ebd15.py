"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen patterns for 2x2x2
    strassen_patterns = [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]},
    ]
    
    def build_decomposition(n_splits, m_splits, p_splits):
        """Build tensor decomposition given splits, return (U, V, W, rank)"""
        n_off = [0]
        for s in n_splits[:-1]:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in m_splits[:-1]:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in p_splits[:-1]:
            p_off.append(p_off[-1] + s)
        
        max_rank = n * m * p
        U = [[0] * max_rank for _ in range(n * m)]
        V = [[0] * max_rank for _ in range(m * p)]
        W = [[0] * max_rank for _ in range(p * n)]
        rank = 0
        
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]
                    
                    i_off = n_off[i_b]
                    j_off = m_off[j_b]
                    k_off = p_off[k_b]
                    
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        for pattern in strassen_patterns:
                            col = rank
                            for li, lj, val in pattern['U']:
                                gi = i_off + li
                                gj = j_off + lj
                                U[gi * m + gj][col] = val
                            for lj, lk, val in pattern['V']:
                                gj = j_off + lj
                                gk = k_off + lk
                                V[gj * p + gk][col] = val
                            for lk, li, val in pattern['W']:
                                gk = k_off + lk
                                gi = i_off + li
                                W[gk * n + gi][col] = val
                            rank += 1
                    else:
                        for li in range(n_b):
                            for lj in range(m_b):
                                for lk in range(p_b):
                                    col = rank
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    rank += 1
        
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
        return U, V, W, rank
    
    # Try multiple split strategies and pick the one with lowest rank
    split_strategies = [
        ([2,1], [2,2,1], [2,2,1]),
        ([2,1], [3,2], [3,2]),
        ([2,1], [2,3], [2,3]),
        ([3], [2,2,1], [2,2,1]),
        ([3], [3,2], [3,2]),
        ([2,1], [2,2,1], [3,2]),
        ([2,1], [3,2], [2,2,1]),
    ]
    
    best_rank = n * m * p
    best_result = None
    
    for ns, ms, ps in split_strategies:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = build_decomposition(ns, ms, ps)
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
        except:
            continue
    
    if best_result is None:
        # Fallback to schoolbook
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i*m*p + j*p + k
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
        return {"U": U, "V": V, "W": W}
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
