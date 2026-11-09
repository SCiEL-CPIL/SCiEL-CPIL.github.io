"""Rewritten program for spectral graph lifts task."""
import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    n = graph['n']
    edges = graph['edges']
    E = len(edges)
    
    # Parse graph structure: two m-cycles + matching
    m = n // 2
    # Build adjacency for base graph
    # We need to identify which edges are cycle edges and which are matching edges
    
    # Let's figure out the edge types
    # Cycle 1: vertices 0..m-1, edges (i, (i+1)%m)
    # Cycle 2: vertices m..2m-1, edges (m+i, m+(i+1)%m)
    # Matching: (i, i+m) for i in 0..m-1
    
    # Build a set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    # Classify edges
    cycle1_edges = []
    cycle2_edges = []
    matching_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            cycle1_edges.append(idx)
        elif u >= m and v >= m:
            cycle2_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    # Helper to compute spectral radius of signed adjacency matrix
    def compute_radius(signs):
        """Compute max |eigenvalue| of signed adjacency matrix."""
        A = np.zeros((n, n), dtype=float)
        for idx, (u, v) in enumerate(edges):
            A[u, v] = signs[idx]
            A[v, u] = signs[idx]
        # Use eigvalsh for symmetric matrix
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    # Helper to check if lift is valid (connected and 3-regular)
    def is_valid_lift(signs):
        """
        Check if the 2-lift defined by signs is connected and 3-regular.
        Lift vertices: (v, 0) and (v, 1) for v in 0..n-1
        +1 edge (u,v): connects (u,0)-(v,0) and (u,1)-(v,1)
        -1 edge (u,v): connects (u,0)-(v,1) and (u,1)-(v,0)
        """
        # Build lift adjacency
        lift_n = 2 * n
        # Count degrees first - should all be 3 if base is 3-regular
        # Base graph: each vertex has degree 3 (2 cycle + 1 matching)
        # So lift is automatically 3-regular if base is 3-regular
        
        # Check connectivity using BFS
        visited = [False] * lift_n
        # Start from (0, 0)
        visited[0] = True
        queue = [0]
        
        while queue:
            node = queue.pop()
            v = node // 2
            side = node % 2
            
            for idx, (u, w) in enumerate(edges):
                if u == v:
                    other = w
                    s = signs[idx]
                    # Determine which side the other endpoint is on
                    if s == 1:
                        other_side = side
                    else:
                        other_side = 1 - side
                    other_node = other * 2 + other_side
                    if not visited[other_node]:
                        visited[other_node] = True
                        queue.append(other_node)
                elif w == v:
                    other = u
                    s = signs[idx]
                    if s == 1:
                        other_side = side
                    else:
                        other_side = 1 - side
                    other_node = other * 2 + other_side
                    if not visited[other_node]:
                        visited[other_node] = True
                        queue.append(other_node)
        
        return all(visited)
    
    # Random sampling with local search
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = np.inf
    
    def try_signs(signs):
        nonlocal best_signs, best_radius
        if not is_valid_lift(signs):
            return False
        r = compute_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
        return True
    
    # Phase 1: Random sampling
    num_random = 2000
    for i in range(num_random):
        if time.time() > deadline - 10:
            break
        signs = rng.choice([-1, 1], size=E).astype(np.int8)
        try_signs(signs)
    
    # Phase 2: Local search from best found and random starts
    def local_search(signs, max_iters=1000):
        """Hill climbing: flip one sign at a time if it improves radius."""
        current = signs.copy()
        current_r = compute_radius(current)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 2:
                break
            improved = False
            # Try flipping each edge
            best_flip_idx = -1
            best_flip_r = current_r
            
            for idx in range(E):
                if time.time() > deadline - 2:
                    break
                trial = current.copy()
                trial[idx] = -trial[idx]
                if not is_valid_lift(trial):
                    continue
                r = compute_radius(trial)
                if r < best_flip_r:
                    best_flip_r = r
                    best_flip_idx = idx
            
            if best_flip_idx >= 0:
                current[best_flip_idx] = -current[best_flip_idx]
                current_r = best_flip_r
                improved = True
            iters += 1
        
        return current, current_r
    
    # Local search from best found
    if best_signs is not None:
        if time.time() < deadline - 5:
            ls_signs, ls_r = local_search(best_signs, max_iters=500)
            if ls_r < best_radius:
                best_radius = ls_r
                best_signs = ls_signs
    
    # Local search from multiple random starts
    num_ls_starts = 50
    for i in range(num_ls_starts):
        if time.time() > deadline - 5:
            break
        signs = rng.choice([-1, 1], size=E).astype(np.int8)
        if not is_valid_lift(signs):
            continue
        ls_signs, ls_r = local_search(signs, max_iters=300)
        if ls_r < best_radius:
            best_radius = ls_r
            best_signs = ls_signs
    
    # Phase 3: More aggressive local search with simulated annealing-like approach
    if best_signs is not None and time.time() < deadline - 3:
        # Continue local search from best
        ls_signs, ls_r = local_search(best_signs, max_iters=2000)
        if ls_r < best_radius:
            best_radius = ls_r
            best_signs = ls_signs
    
    if best_signs is None:
        # Fallback: return all +1 (should be valid for prism graphs)
        best_signs = np.ones(E, dtype=np.int8)
    
    return best_signs
