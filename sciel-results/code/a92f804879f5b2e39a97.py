import numpy as np
import time
from collections import deque

def _is_lift_connected(signs, n, edges):
    """Check if the 2-lift is connected using union-find."""
    parent = list(range(2 * n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            union(2*u, 2*v)
            union(2*u+1, 2*v+1)
        else:
            union(2*u, 2*v+1)
            union(2*u+1, 2*v)
    
    root = find(0)
    return all(find(v) == root for v in range(2 * n))

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Build base adjacency matrix
    A_base = np.zeros((n, n), dtype=float)
    for (u, v) in edges:
        A_base[u, v] = 1.0
        A_base[v, u] = 1.0

    # Identify the matching edges (i, i+m) for i in 0..m-1
    # These are the "vertical" edges in the prism graph
    matching_edge_idx = {}
    for i, (u, v) in enumerate(edges):
        if v == u + m or u == v + m:
            matching_edge_idx[i] = (min(u, v), max(u, v))

    # The other edges are cycle edges (top and bottom cycles)
    cycle_edge_indices = [i for i in range(num_edges) if i not in matching_edge_idx]
    matching_edge_indices = list(matching_edge_idx.keys())
    num_matching = len(matching_edge_indices)

    # For the prism graph, the matching edges connect i to i+m for i=0..m-1
    # We'll parameterize signs on matching edges by vertex potentials
    # But first, let's do a direct search over all sign assignments

    best_radius = float('inf')
    best_signs = None

    # Phase 1: Exhaustive search over matching edge signs (2^m possibilities)
    # Fix cycle edges to +1, vary matching edges
    # For m=4,5,7,9 this is 16, 32, 128, 512 possibilities
    total_masks = 1 << num_matching
    for mask in range(1, total_masks):
        if time.time() > deadline:
            break
        
        # Build signs array: cycle edges = +1, matching edges depend on mask
        signs = np.ones(num_edges, dtype=int)
        for k, edge_idx in enumerate(matching_edge_indices):
            if mask & (1 << k):
                signs[edge_idx] = -1
        
        # Build signed adjacency matrix
        A = A_base.copy()
        for i, (u, v) in enumerate(edges):
            if signs[i] == -1:
                A[u, v] = -1.0
                A[v, u] = -1.0
        
        # Check connectivity
        if not _is_lift_connected(signs, n, edges):
            continue
        
        eigvals = np.linalg.eigvalsh(A)
        r = float(np.max(np.abs(eigvals)))
        
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    # Phase 2: Full random search with simulated annealing
    if best_signs is not None and time.time() < deadline:
        rng = np.random.default_rng(42)
        current_signs = best_signs.copy()
        current_radius = best_radius
        best_signs_sa = best_signs.copy()
        best_radius_sa = best_radius
        
        temperature = 0.1
        cooling = 0.9995
        
        attempts = 0
        while time.time() < deadline and temperature > 0.001:
            attempts += 1
            
            # Pick a random edge to flip
            edge_idx = rng.integers(0, num_edges)
            trial_signs = current_signs.copy()
            trial_signs[edge_idx] = -trial_signs[edge_idx]
            
            # Check connectivity
            if not _is_lift_connected(trial_signs, n, edges):
                continue
            
            # Compute spectral radius
            A = A_base.copy()
            for i, (u, v) in enumerate(edges):
                if trial_signs[i] == -1:
                    A[u, v] = -1.0
                    A[v, u] = -1.0
            
            eigvals = np.linalg.eigvalsh(A)
            trial_radius = float(np.max(np.abs(eigvals)))
            
            delta = trial_radius - current_radius
            
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                current_signs = trial_signs
                current_radius = trial_radius
                
                if trial_radius < best_radius_sa:
                    best_radius_sa = trial_radius
                    best_signs_sa = trial_signs.copy()
            
            temperature *= cooling
        
        if best_radius_sa < best_radius:
            best_radius = best_radius_sa
            best_signs = best_signs_sa

    # Phase 3: Local search polish
    if best_signs is not None and time.time() < deadline:
        current_signs = best_signs.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                trial_signs = current_signs.copy()
                trial_signs[i] = -trial_signs[i]
                
                if not _is_lift_connected(trial_signs, n, edges):
                    continue
                
                A = A_base.copy()
                for j, (u, v) in enumerate(edges):
                    if trial_signs[j] == -1:
                        A[u, v] = -1.0
                        A[v, u] = -1.0
                
                eigvals = np.linalg.eigvalsh(A)
                r = float(np.max(np.abs(eigvals)))
                
                if r < best_radius - 1e-12:
                    best_radius = r
                    current_signs = trial_signs
                    best_signs = trial_signs.copy()
                    improved = True

    # If we found a solution, use it
    if best_signs is not None:
        return best_signs

    # Fallback: all +1 except one matching edge
    signs = np.ones(num_edges, dtype=int)
    if matching_edge_indices:
        signs[matching_edge_indices[0]] = -1

    return signs
