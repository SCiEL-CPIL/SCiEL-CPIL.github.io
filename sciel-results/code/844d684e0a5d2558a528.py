import time
import numpy as np

def construct():
    """
    Return a rank-117 decomposition of the 5x5x5 matrix multiplication tensor.
    
    Strategy: Partition indices into G0={0,1}, G1={2,3}, G2={4}.
    Apply Strassen's 7-term algorithm to the 2x2x2 blocks.
    Use schoolbook multiplication for all other blocks.
    """
    n, m, p = 5, 5, 5
    
    G0 = [0, 1]
    G1 = [2, 3]
    G2 = [4]
    groups = [G0, G1, G2]
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for idx, val in u_entries:
            u[idx] = val
        for idx, val in v_entries:
            v[idx] = val
        for idx, val in w_entries:
            w[idx] = val
        
        terms.append((u, v, w))
    
    for i_group in groups:
        for j_group in groups:
            for k_group in groups:
                if len(i_group) == 2 and len(j_group) == 2 and len(k_group) == 2:
                    _add_strassen_block(i_group, j_group, k_group, add_term, n, m, p)
                else:
                    # Schoolbook multiplication
                    for i in i_group:
                        for j in j_group:
                            for k in k_group:
                                add_term([(i * m + j, 1)],
                                         [(j * p + k, 1)],
                                         [(k * n + i, 1)])
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            U[idx][r] = u[idx]
        for idx in range(m * p):
            V[idx][r] = v[idx]
        for idx in range(p * n):
            W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}


def _add_strassen_block(i_group, j_group, k_group, add_term, n, m, p):
    """
    Add Strassen terms for a specific 2x2x2 block.
    """
    i_map = i_group
    j_map = j_group
    k_map = k_group
    
    def map_idx(group, local_idx):
        return group[local_idx]
    
    # Strassen's 7-term algorithm for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Term 1: M1
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 1)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1), (map_idx(k_map, 1)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 2: M2
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 3: M3
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 1)*p + map_idx(k_map, 1), -1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1)]
    add_term(u, v, w)
    
    # Term 4: M4
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 1), 1), (map_idx(j_map, 0)*p + map_idx(k_map, 0), -1)]
    w = [(map_idx(k_map, 1)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 5: M5
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 0)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 1)*n + map_idx(i_map, 0), 1)]
    add_term(u, v, w)
    
    # Term 6: M6
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 0)*m + map_idx(j_map, 0), -1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 0)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 7: M7
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 1), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 1), -1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 1)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1)]
    add_term(u, v, w)


def _add_strassen_block(i_group, j_group, k_group, add_term, n, m, p):
    """
    Add Strassen terms for a specific 2x2x2 block.
    """
    i_map = i_group
    j_map = j_group
    k_map = k_group
    
    def map_idx(group, local_idx):
        return group[local_idx]
    
    # Term 1
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 1)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1), (map_idx(k_map, 1)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 2
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 1), 1), (map_idx(k_map, 1)*n + map_idx(i_map, 1), -1)]
    add_term(u, v, w)
    
    # Term 3
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 1), 1), (map_idx(j_map, 1)*p + map_idx(k_map, 1), -1)]
    w = [(map_idx(k_map, 1)*n + map_idx(i_map, 0), 1), (map_idx(k_map, 1)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 4
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 0)*p + map_idx(k_map, 0), -1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1), (map_idx(k_map, 0)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 5
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 0)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), -1), (map_idx(k_map, 1)*n + map_idx(i_map, 0), 1)]
    add_term(u, v, w)
    
    # Term 6
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 0)*m + map_idx(j_map, 0), -1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 0)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 1)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 7
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 1), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 1), -1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 1)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1)]
    add_term(u, v, w)
