import time
import numpy as np
from itertools import product

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy: Local search with multiple restarts to minimize spectral radius.
    """
    deadline = time.time() + 85.0  # Safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build adjacency matrix indices
    # We need to quickly compute the spectral radius of the signed adjacency matrix.
    # Precompute the structure to speed up evaluation.
    # A is 0/1. S is the signed matrix.
    # S[u,v] = sign[i] if (u,v) is edge i.
    
    # Precompute edge indices
    edge_index_map = {}
    for i, (u, v) in enumerate(edges):
        edge_index_map[(u, v)] = i
        edge_index_map[(v, u)] = i
        
    # Helper to build signed adjacency matrix from signs
    def build_signed_matrix(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            val = float(signs[i])
            S[u, v] = val
            S[v, u] = val
        return S

    # Helper to compute spectral radius
    def get_spectral_radius(signs):
        S = build_signed_matrix(signs)
        # Use eigvalsh for symmetric matrices
        try:
            eigvals = np.linalg.eigvalsh(S)
            return np.max(np.abs(eigvals))
        except np.linalg.LinAlgError:
            return np.inf

    best_signs = None
    best_radius = np.inf
    
    # Initializations:
    # 1. All +1
    # 2. All -1
    # 3. Random restarts
    
    initializations = []
    initializations.append(np.ones(num_edges, dtype=np.int8))
    initializations.append(-np.ones(num_edges, dtype=np.int8))
    
    # Add some random initializations
    rng = np.random.default_rng(42)
    for _ in range(10):
        initializations.append(rng.choice([-1, 1], size=num_edges).astype(np.int8))
        
    # Local search function
    def local_search(start_signs):
        current_signs = start_signs.copy()
        current_radius = get_spectral_radius(current_signs)
        improved = True
        while improved:
            improved = False
            for i in range(num_edges):
                # Try flipping sign i
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                new_radius = get_spectral_radius(current_signs)
                
                if new_radius < current_radius - 1e-9:
                    current_radius = new_radius
                    improved = True
                else:
                    # Revert
                    current_signs[i] = old_sign
                    
        return current_signs, current_radius

    for init in initializations:
        if time.time() > deadline:
            break
            
        res_signs, res_radius = local_search(init)
        
        if res_radius < best_radius:
            best_radius = res_radius
            best_signs = res_signs.copy()
            
        # If we have time, do more random restarts
        if time.time() < deadline - 5.0:
             # Do a few more random restarts in a loop
             while time.time() < deadline - 5.0:
                 random_init = rng.choice([-1, 1], size=num_edges).astype(np.int8)
                 res_signs, res_radius = local_search(random_init)
                 if res_radius < best_radius:
                     best_radius = res_radius
                     best_signs = res_signs.copy()
                 # If we've hit a very low radius, we might stop early?
                 # Theoretical min for 3-regular is 3.0.
                 if best_radius < 3.01:
                     break
             if best_radius < 3.01:
                 break

    if best_signs is None:
        # Fallback to all +1 if something went wrong
        best_signs = np.ones(num_edges, dtype=np.int8)
        
    return best_signs
