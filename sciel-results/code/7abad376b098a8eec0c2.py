"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    deadline = time.time() + 55.0
    
    # Identify edge types in the prism graph
    edge_types = []
    edge_indices_by_type = {0: [], 1: [], 2: []}
    
    for idx, (u, v) in enumerate(edges):
        u_min = min(u, v)
        u_max = max(u, v)
        if u_min < m and u_max < m:
            edge_types.append(0)
            edge_indices_by_type[0].append(idx)
        elif u_min >= m and u_max >= m:
            edge_types.append(1)
            edge_indices_by_type[1].append(idx)
        else:
            edge_types.append(2)
            edge_indices_by_type[2].append(idx)
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] += s
            A[v, u] += s
        return A
    
    def spectral_radius(signs):
        A = build_signed_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        try:
            radius, details = check_lift(graph, signs)
            return True, radius
        except:
            return False, None
    
    best_signs = None
    best_radius = float('inf')
    
    cycle2_count = len(edge_indices_by_type[1])
    matching_count = len(edge_indices_by_type[2])
    
    # Gauge reduction: fix cycle1 to +1, and fix first matching edge to +1
    # Search over cycle2 (m bits) and remaining matching (m-1 bits)
    # Total: 2^(2m-1)
    
    if m <= 9:
        fixed_matching_idx = edge_indices_by_type[2][0]
        free_matching_indices = edge_indices_by_type[2][1:]
        
        for cycle2_bits in product([1, -1], repeat=cycle2_count):
            for matching_bits in product([1, -1], repeat=len(free_matching_indices)):
                if time.time() > deadline:
                    break
                
                signs = np.ones(num_edges, dtype=np.int8)
                
                for i, idx in enumerate(edge_indices_by_type[1]):
                    signs[idx] = cycle2_bits[i]
                
                signs[fixed_matching_idx] = 1
                for i, idx in enumerate(free_matching_indices):
                    signs[idx] = matching_bits[i]
                
                valid, radius = is_valid_lift(signs)
                if valid and radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
    else:
        rng = np.random.default_rng(3)
        for _ in range(min(budget, 10000)):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
            valid, radius = is_valid_lift(signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
    
    return best_signs
# EVOLVE-BLOCK-END
