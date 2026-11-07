"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    deadline = time.time() + 85.0
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Build adjacency structure for the base graph
    # edges: two m-cycles (0..m-1 and m..2m-1) plus matching (i, i+m)
    # Edge order: we need to figure out the exact ordering
    # Let's just use the given edges list
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_matrix(signs)
        evals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(evals)))
    
    def is_connected_lift(signs):
        """Check if the 2-lift is connected using BFS on the lift graph."""
        # The lift has 2n vertices: (v, 0) and (v, 1) for v in 0..n-1
        # For each edge (u, v) with sign s:
        #   if s = +1: (u,0)-(v,0) and (u,1)-(v,1)
        #   if s = -1: (u,0)-(v,1) and (u,1)-(v,0)
        
        # Build adjacency list for the lift
        lift_adj = [[] for _ in range(2 * n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                lift_adj[u].append(v)
                lift_adj[v].append(u)
                lift_adj[n + u].append(n + v)
                lift_adj[n + v].append(n + u)
            else:
                lift_adj[u].append(n + v)
                lift_adj[n + v].append(u)
                lift_adj[n + u].append(v)
                lift_adj[v].append(n + u)
        
        # BFS from (0, 0) which is vertex 0 in the lift
        visited = [False] * (2 * n)
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        return count == 2 * n
    
    def is_valid(signs):
        if not is_connected_lift(signs):
            return False
        # Check 3-regularity: each vertex in the lift must have degree 3
        # In the base graph, each vertex has degree 3 (2 cycle edges + 1 matching edge)
        # In the lift, each vertex (v, s) connects to 3 neighbors (one per base edge)
        # So it's automatically 3-regular if the base is 3-regular
        # The base prism graph is 3-regular, so the lift is automatically 3-regular
        return True
    
    # Initial random sampling
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = float('inf')
    
    # Sample budget random sign vectors
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges)
        if is_valid(signs):
            r = spectral_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1 should be valid (connected, 3-regular)
        best_signs = np.ones(num_edges, dtype=int)
        best_radius = spectral_radius(best_signs)
    
    # Local search: flip individual edges
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    while time.time() < deadline:
        improved = False
        # Try flipping each edge
        for i in range(num_edges):
            if time.time() > deadline - 5:
                break
            old_sign = current_signs[i]
            current_signs[i] = -old_sign
            if is_valid(current_signs):
                new_radius = spectral_radius(current_signs)
                if new_radius < current_radius - 1e-12:
                    current_radius = new_radius
                    improved = True
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current_signs.copy()
                else:
                    current_signs[i] = old_sign
            else:
                current_signs[i] = old_sign
        if not improved:
            # Try random restarts from best
            if time.time() > deadline - 10:
                break
            # Perturb current best
            perturb = rng.choice([-1, 1], size=num_edges)
            num_flips = rng.integers(1, min(5, num_edges) + 1)
            flip_indices = rng.choice(num_edges, size=num_flips, replace=False)
            new_signs = best_signs.copy()
            new_signs[flip_indices] *= perturb[flip_indices]
            if is_valid(new_signs):
                new_radius = spectral_radius(new_signs)
                if new_radius < current_radius:
                    current_signs = new_signs.copy()
                    current_radius = new_radius
                else:
                    # Continue local search from perturbed state even if worse? No, reset
                    pass
            # If no improvement after perturbation, break to save time
            if time.time() > deadline - 5:
                break
    
    return best_signs.astype(np.int8)
# EVOLVE-BLOCK-END
