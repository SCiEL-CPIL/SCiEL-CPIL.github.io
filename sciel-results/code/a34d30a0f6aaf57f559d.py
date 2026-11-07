"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def sign_edges(graph, budget):
    deadline = time.time() + 85.0
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
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
        # Build adjacency of the lift
        # Vertices in lift: (v, 0) and (v, 1) for v in 0..n-1
        # Edge (u,v) with sign +1: (u,0)-(v,0) and (u,1)-(v,1)
        # Edge (u,v) with sign -1: (u,0)-(v,1) and (u,1)-(v,0)
        
        lift_n = 2 * n
        adj = [[] for _ in range(lift_n)]
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                # +1: same side
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                # -1: opposite side
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)
        
        # BFS from vertex 0
        visited = [False] * lift_n
        visited[0] = True
        queue = [0]
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        return all(visited)
    
    def is_valid(signs):
        if not is_connected_lift(signs):
            return False
        # Check 3-regular: each vertex in lift has degree 3
        # Since base graph is 3-regular and lift preserves degree, this should hold
        # But let's verify
        lift_n = 2 * n
        degrees = [0] * lift_n
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                degrees[u * 2] += 1
                degrees[v * 2] += 1
                degrees[u * 2 + 1] += 1
                degrees[v * 2 + 1] += 1
            else:
                degrees[u * 2] += 1
                degrees[v * 2 + 1] += 1
                degrees[u * 2 + 1] += 1
                degrees[v * 2] += 1
        if any(d != 3 for d in degrees):
            return False
        return True
    
    # Strategy: 
    # 1. Start with a random valid signing
    # 2. Use local search to minimize spectral radius
    # 3. Try multiple restarts
    
    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(3)
    
    def random_valid_signing():
        for _ in range(10000):
            signs = rng.choice([-1, 1], size=num_edges)
            if is_valid(signs):
                return signs
        return None
    
    # First, find an initial valid signing
    initial = random_valid_signing()
    if initial is None:
        # Fallback: try all +1 (should be valid for prism graphs)
        initial = np.ones(num_edges, dtype=int)
        if not is_valid(initial):
            # Try alternating patterns
            for pattern in range(4):
                signs = np.ones(num_edges, dtype=int)
                for i in range(num_edges):
                    if (i + pattern) % 2 == 0:
                        signs[i] = -1
                if is_valid(signs):
                    initial = signs
                    break
    
    best_signs = initial.copy()
    best_radius = spectral_radius(best_signs)
    
    # Local search: flip individual edges and check improvement
    def local_search(signs, max_iter=10000):
        nonlocal best_signs, best_radius
        current = signs.copy()
        current_radius = spectral_radius(current)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try flipping each edge
            for i in range(num_edges):
                new_signs = current.copy()
                new_signs[i] = -new_signs[i]
                
                if not is_valid(new_signs):
                    continue
                
                new_radius = spectral_radius(new_signs)
                if new_radius < current_radius - 1e-10:
                    current = new_signs
                    current_radius = new_radius
                    improved = True
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = current.copy()
                    break
            
            if not improved:
                break
        
        return current
    
    # Run local search from initial
    result = local_search(best_signs)
    if spectral_radius(result) < best_radius:
        best_signs = result
        best_radius = spectral_radius(best_signs)
    
    # Try multiple random restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        restart_signs = random_valid_signing()
        if restart_signs is None:
            continue
        
        # Local search from this restart
        result = local_search(restart_signs, max_iter=5000)
        r_radius = spectral_radius(result)
        if r_radius < best_radius:
            best_signs = result
            best_radius = r_radius
    
    # Final polish: more aggressive local search
    if time.time() < deadline - 15:
        result = local_search(best_signs, max_iter=50000)
        r_radius = spectral_radius(result)
        if r_radius < best_radius:
            best_signs = result
            best_radius = r_radius
    
    return best_signs
# EVOLVE-BLOCK-END
