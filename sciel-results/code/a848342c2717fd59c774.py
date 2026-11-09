import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def is_valid_fast(n, adj_bits):
        """Fast validation using precomputed adjacency bitmasks."""
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj_bits[i] >> j) & 1:
                    continue
                common_ij = adj_bits[i] & adj_bits[j]
                # Find common neighbors > j
                k_mask = common_ij & ~((1 << (j + 1)) - 1)
                while k_mask:
                    k_bit = k_mask & -k_mask
                    k = k_bit.bit_length() - 1
                    k_mask ^= k_bit
                    common_ijk = adj_bits[i] & adj_bits[j] & adj_bits[k]
                    l_mask = common_ijk & ~((1 << (k + 1)) - 1)
                    if l_mask:
                        return False
                        
        # Check Independent Set of 6 using complement
        # comp_bits[i] has bit j set if i and j are NOT adjacent (and i != j)
        comp_bits = []
        all_mask = (1 << n) - 1
        for i in range(n):
            comp_bits.append(all_mask ^ adj_bits[i] ^ (1 << i))
            
        for i in range(n):
            for j in range(i + 1, n):
                if not (comp_bits[i] >> j) & 1:
                    continue
                common_ij = comp_bits[i] & comp_bits[j]
                k_mask = common_ij & ~((1 << (j + 1)) - 1)
                while k_mask:
                    k_bit = k_mask & -k_mask
                    k = k_bit.bit_length() - 1
                    k_mask ^= k_bit
                    common_ijk = comp_bits[i] & comp_bits[j] & comp_bits[k]
                    l_mask = common_ijk & ~((1 << (k + 1)) - 1)
                    while l_mask:
                        l_bit = l_mask & -l_mask
                        l = l_bit.bit_length() - 1
                        l_mask ^= l_bit
                        common_ijkl = comp_bits[i] & comp_bits[j] & comp_bits[k] & comp_bits[l]
                        m_mask = common_ijkl & ~((1 << (l + 1)) - 1)
                        while m_mask:
                            m_bit = m_mask & -m_mask
                            m = m_bit.bit_length() - 1
                            m_mask ^= m_bit
                            common_ijklm = comp_bits[i] & comp_bits[j] & comp_bits[k] & comp_bits[l] & comp_bits[m]
                            p_mask = common_ijklm & ~((1 << (m + 1)) - 1)
                            if p_mask:
                                return False
                                
        return True

    def build_adj_bits(n, edges):
        adj_bits = [0] * n
        for u, v in edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        return adj_bits

    # Known 35-vertex construction: complement of line graph of K_7
    # Vertices are 2-subsets of {0,1,...,6}, two vertices adjacent iff they share exactly one element
    # This gives the complement of the line graph of K_7
    vertices = list(combinations(range(7), 2))
    n = len(vertices)  # C(7,2) = 21... wait, that's only 21
    
    # Actually, let's use the standard 35-vertex Ramsey(4,6) graph
    # It's a strongly regular graph with parameters (35, 16, 7, 8)
    # Construction: vertices are 3-subsets of {0,1,...,6}, two adjacent iff they share exactly 1 element
    vertices = list(combinations(range(7), 3))
    n = len(vertices)  # C(7,3) = 35
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            u = vertices[i]
            v = vertices[j]
            inter_size = len(set(u).intersection(v))
            if inter_size == 1:
                edges.append([i, j])
                
    adj_bits = build_adj_bits(n, edges)
    if is_valid_fast(n, adj_bits):
        return {'n': n, 'edges': edges}

    # Fallback: try smaller constructions
    # Paley graph of order 17
    q = 17
    squares = {x * x % q for x in range(1, q)}
    edges_17 = [[i, j] for i in range(q) for j in range(i + 1, q) if (j - i) % q in squares]
    adj_bits_17 = build_adj_bits(q, edges_17)
    if is_valid_fast(q, adj_bits_17):
        return {'n': q, 'edges': edges_17}
        
    return {'n': 1, 'edges': []}
