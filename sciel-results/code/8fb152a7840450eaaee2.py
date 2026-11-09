import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def is_valid(n, edges):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u, v] = True
            adj[v, u] = True
            
        # Check K4 using bitmask
        adj_bits = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    adj_bits[i] |= (1 << j)
                    
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i, j]: continue
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
        comp = ~adj
        np.fill_diagonal(comp, False)
        
        comp_bits = [0] * n
        for i in range(n):
            for j in range(n):
                if comp[i, j]:
                    comp_bits[i] |= (1 << j)
                    
        for i in range(n):
            for j in range(i + 1, n):
                if not comp[i, j]: continue
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

    # First, try the known 35-vertex strongly regular graph
    vertices = list(combinations(range(7), 3))
    n = len(vertices)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            u = vertices[i]
            v = vertices[j]
            inter_size = len(set(u).intersection(v))
            if inter_size != 1:
                edges.append([i, j])
                
    if is_valid(n, edges):
        return {'n': n, 'edges': edges}

    # Try randomized algebraic construction using finite field geometry
    # Use GF(q) with q = p^k for prime p
    def construct_finite_field_graph(q, max_attempts=50):
        if time.time() > deadline - 10:
            return None
            
        # Try different multiplicative character constructions
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Use GF(q) where q is a prime power
            # For simplicity, use prime q
            if q > 40:
                continue
                
            # Generate random quadratic form coefficients
            a = random.randint(1, q - 1)
            b = random.randint(1, q - 1)
            c = random.randint(1, q - 1)
            
            # Define graph: vertices are 0..q-1, edge (i,j) if quadratic form condition met
            # Use: a*i^2 + b*j^2 + c*i*j != 0 mod q
            adj = np.zeros((q, q), dtype=bool)
            edges_ff = []
            
            for i in range(q):
                for j in range(i + 1, q):
                    # Quadratic form: a*i^2 + b*j^2 + c*i*j mod q
                    val = (a * i * i + b * j * j + c * i * j) % q
                    if val != 0:
                        adj[i, j] = True
                        adj[j, i] = True
                        edges_ff.append([i, j])
                        
            if is_valid(q, edges_ff):
                return {'n': q, 'edges': edges_ff}
                
        return None

    # Try various prime sizes
    for q in [17, 19, 23, 29, 31, 37]:
        if time.time() > deadline - 15:
            break
        result = construct_finite_field_graph(q)
        if result and result['n'] > 17:
            return result

    # Fallback: Paley graph of order 17
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    if is_valid(17, edges_17):
        return {'n': 17, 'edges': edges_17}
        
    return {'n': 1, 'edges': []}
