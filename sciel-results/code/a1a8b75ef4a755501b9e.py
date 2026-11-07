"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def sign_edges(graph, budget):
    """
    Returns sign array of shape (num_edges,) with values -1 or +1 for a prism graph.
    Minimizes the spectral radius of the signed adjacency matrix of the base graph.
    """
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    m = graph['n'] // 2
    n = graph['n']  # 2m
    edges = graph['edges']
    num_edges = len(edges)

    # Build adjacency structure
    # The prism graph has:
    # - Top cycle: (0,1), (1,2), ..., (m-1, 0)
    # - Bottom cycle: (m, m+1), ..., (2m-1, m)
    # - Matching: (i, i+m) for i in 0..m-1
    # Total edges: 3m

    # Let's identify edge types
    # For each edge, determine if it's top cycle, bottom cycle, or matching
    # Top cycle edges: both endpoints in [0, m-1]
    # Bottom cycle edges: both endpoints in [m, 2m-1]
    # Matching edges: one in [0, m-1], one in [m, 2m-1]

    # We'll work with the sign vector directly
    # signs[i] is the sign for edges[i]

    # Build the signed adjacency matrix given signs
    def build_signed_adj(signs):
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] += s
            A[v, u] += s
        return A

    def spectral_radius(signs):
        A = build_signed_adj(signs)
        # A is symmetric, use eigvalsh
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def is_valid_lift(signs):
        """Check if the 2-lift is connected and 3-regular."""
        # The lift has 2n vertices: (v, 0) and (v, 1) for v in 0..n-1
        # +1 joins equal sides: (u,0)-(v,0) and (u,1)-(v,1)
        # -1 joins opposite sides: (u,0)-(v,1) and (u,1)-(v,0)
        
        # Build adjacency for the lift
        lift_n = 2 * n
        lift_adj = np.zeros((lift_n, lift_n), dtype=int)
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                # Equal sides
                lift_adj[u*2, v*2] += 1
                lift_adj[v*2, u*2] += 1
                lift_adj[u*2+1, v*2+1] += 1
                lift_adj[v*2+1, u*2+1] += 1
            else:
                # Opposite sides
                lift_adj[u*2, v*2+1] += 1
                lift_adj[v*2+1, u*2] += 1
                lift_adj[u*2+1, v*2] += 1
                lift_adj[v*2, u*2+1] += 1
        
        # Check 3-regular: each vertex should have degree 3
        degrees = np.sum(lift_adj, axis=1)
        if not np.all(degrees == 3):
            return False
        
        # Check connected: BFS from vertex 0
        visited = np.zeros(lift_n, dtype=bool)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            u = queue.pop(0)
            for v in range(lift_n):
                if lift_adj[u, v] > 0 and not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        
        return count == lift_n

    # Initial solution: try random signs
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = np.inf
    
    # Try some initial random solutions
    for _ in range(50):
        signs = rng.choice([-1, 1], size=num_edges)
        if is_valid_lift(signs):
            r = spectral_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=int)
        best_radius = spectral_radius(best_signs)
    
    # Local search: flip one sign at a time
    def local_search(signs, current_radius, max_iters=10000):
        signs = signs.copy()
        current_r = current_radius
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            for i in range(num_edges):
                # Try flipping sign i
                old_sign = signs[i]
                signs[i] = -old_sign
                if is_valid_lift(signs):
                    r = spectral_radius(signs)
                    if r < current_r - 1e-10:
                        current_r = r
                        improved = True
                    else:
                        signs[i] = old_sign
                else:
                    signs[i] = old_sign
        return signs, current_r
    
    # Run local search from best initial solution
    best_signs, best_radius = local_search(best_signs, best_radius)
    
    # Try multiple random restarts if time permits
    num_restarts = 0
    while time.time() - start_time < deadline - 5:
        num_restarts += 1
        signs = rng.choice([-1, 1], size=num_edges)
        if not is_valid_lift(signs):
            continue
        r = spectral_radius(signs)
        if r < best_radius:
            best_signs = signs.copy()
            best_radius = r
            best_signs, best_radius = local_search(best_signs, best_radius)
    
    # If still time, try more aggressive search
    if time.time() - start_time < deadline - 2:
        # Try flipping pairs of signs
        for i in range(num_edges):
            for j in range(i+1, num_edges):
                if time.time() - start_time > deadline - 1:
                    break
                old_si, old_sj = best_signs[i], best_signs[j]
                best_signs[i] = -old_si
                best_signs[j] = -old_sj
                if is_valid_lift(best_signs):
                    r = spectral_radius(best_signs)
                    if r < best_radius - 1e-10:
                        best_radius = r
                    else:
                        best_signs[i] = old_si
                        best_signs[j] = old_sj
                else:
                    best_signs[i] = old_si
                    best_signs[j] = old_sj
    
    return best_signs.astype(int)

# EVOLVE-BLOCK-END
