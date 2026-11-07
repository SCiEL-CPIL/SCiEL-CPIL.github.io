"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge endpoints as arrays for fast matrix construction
    edge_u = np.array([e[0] for e in edges], dtype=int)
    edge_v = np.array([e[1] for e in edges], dtype=int)
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        return A
    
    def spectral_radius(signs):
        A = build_signed_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        # Check if the 2-lift is connected
        # Build the lift graph with 2n vertices
        lift_adj = [[] for _ in range(2 * n)]
        for i in range(num_edges):
            u, v = edges[i]
            s = signs[i]
            for l in [0, 1]:
                l2 = l if s == 1 else 1 - l
                lift_adj[u * 2 + l].append(v * 2 + l2)
                lift_adj[v * 2 + l2].append(u * 2 + l)
        
        # BFS from (0, 0)
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        while queue:
            node = queue.pop(0)
            for nb in lift_adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        return all(visited)
    
    def is_valid(signs):
        return is_connected_lift(signs)
    
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    
    # Simulated annealing with multiple restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial signs
        signs = rng.choice([-1, 1], size=num_edges)
        
        # Ensure valid start
        if not is_valid(signs):
            valid = False
            for _ in range(200):
                if time.time() > deadline:
                    break
                flip_idx = rng.integers(num_edges)
                signs[flip_idx] *= -1
                if is_valid(signs):
                    valid = True
                    break
            if not valid:
                continue
        
        cur_radius = spectral_radius(signs)
        if cur_radius < best_radius:
            best_radius = cur_radius
            best_signs = signs.copy()
        
        # Simulated annealing
        T_start = 0.1
        T_end = 0.001
        num_steps = 2000
        step = 0
        
        for step in range(num_steps):
            if time.time() > deadline:
                break
            
            T = T_start * (T_end / T_start) ** (step / num_steps)
            
            # Pick a random edge to flip
            idx = rng.integers(num_edges)
            old_sign = signs[idx]
            signs[idx] = -old_sign
            
            if is_valid(signs):
                new_radius = spectral_radius(signs)
                delta = new_radius - cur_radius
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    cur_radius = new_radius
                    if cur_radius < best_radius:
                        best_radius = cur_radius
                        best_signs = signs.copy()
                else:
                    signs[idx] = old_sign
            else:
                signs[idx] = old_sign
    
    # If no valid solution found, return all +1
    if best_signs is None:
        signs = np.ones(num_edges, dtype=int)
        if not is_valid(signs):
            signs = -np.ones(num_edges, dtype=int)
            if not is_valid(signs):
                for _ in range(1000):
                    signs = rng.choice([-1, 1], size=num_edges)
                    if is_valid(signs):
                        break
        best_signs = signs
    
    return best_signs
# EVOLVE-BLOCK-END
