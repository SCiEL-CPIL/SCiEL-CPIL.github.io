"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    deadline = time.time() + 52
    
    # Precompute edge index pairs
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    # Build adjacency structure for lift connectivity check
    # For each edge, store (u, v)
    
    def build_signed_matrix(signs):
        """Build signed adjacency matrix from signs array."""
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(num_edges):
            s = signs[i]
            u = edge_u[i]
            v = edge_v[i]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(A):
        """Compute spectral radius using eigvalsh."""
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_lift_connected(signs):
        """Check if the 2-lift defined by signs is connected."""
        # Lift has 2n vertices: (v,0) and (v,1)
        # Edge (u,v) with sign +1: (u,0)-(v,0), (u,1)-(v,1)
        # Edge (u,v) with sign -1: (u,0)-(v,1), (u,1)-(v,0)
        lift_n = 2 * n
        adj = [[] for _ in range(lift_n)]
        for i in range(num_edges):
            s = signs[i]
            u = edge_u[i]
            v = edge_v[i]
            if s == 1:
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:  # s == -1
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)
        
        # BFS from vertex 0
        visited = [False] * lift_n
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        return count == lift_n
    
    def evaluate(signs):
        """Evaluate a sign assignment. Returns spectral radius or None if invalid."""
        A = build_signed_matrix(signs)
        radius = spectral_radius(A)
        if not is_lift_connected(signs):
            return None
        return radius
    
    # Initial random sampling
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = float('inf')
    
    # Try some random assignments
    for _ in range(200):
        if time.time() > deadline:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        r = evaluate(signs)
        if r is not None and r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    
    # If no valid found, try all +1 or all -1
    if best_signs is None:
        for trial_signs in [np.ones(num_edges, dtype=np.int8), -np.ones(num_edges, dtype=np.int8)]:
            r = evaluate(trial_signs)
            if r is not None and r < best_radius:
                best_radius = r
                best_signs = trial_signs.copy()
    
    if best_signs is None:
        # Fallback: should not happen, but return something
        return np.ones(num_edges, dtype=np.int8)
    
    # Local search: flip one edge at a time
    signs = best_signs.copy()
    current_radius = best_radius
    
    # Precompute for faster matrix updates
    # We can update the matrix incrementally when flipping one edge
    
    def flip_edge(signs, A, edge_idx):
        """Flip the sign of edge edge_idx and update A in place. Returns new radius."""
        u = edge_u[edge_idx]
        v = edge_v[edge_idx]
        old_sign = signs[edge_idx]
        new_sign = -old_sign
        signs[edge_idx] = new_sign
        A[u, v] = new_sign
        A[v, u] = new_sign
        return new_sign
    
    # Local search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        A = build_signed_matrix(signs)
        
        for i in range(num_edges):
            if time.time() > deadline:
                break
            
            # Flip edge i
            old_sign = signs[i]
            u = edge_u[i]
            v = edge_v[i]
            new_sign = -old_sign
            signs[i] = new_sign
            A[u, v] = new_sign
            A[v, u] = new_sign
            
            # Check connectivity
            if is_lift_connected(signs):
                r = spectral_radius(A)
                if r < current_radius:
                    current_radius = r
                    best_radius = r
                    best_signs = signs.copy()
                    improved = True
                else:
                    # Revert
                    signs[i] = old_sign
                    A[u, v] = old_sign
                    A[v, u] = old_sign
            else:
                # Revert
                signs[i] = old_sign
                A[u, v] = old_sign
                A[v, u] = old_sign
    
    # Multi-start local search with more random restarts
    if time.time() < deadline - 5:
        for restart in range(50):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
            r = evaluate(signs)
            if r is None:
                continue
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
            
            # Local search from this point
            current_radius = r
            improved = True
            while improved and time.time() < deadline:
                improved = False
                A = build_signed_matrix(signs)
                for i in range(num_edges):
                    if time.time() > deadline:
                        break
                    old_sign = signs[i]
                    u = edge_u[i]
                    v = edge_v[i]
                    new_sign = -old_sign
                    signs[i] = new_sign
                    A[u, v] = new_sign
                    A[v, u] = new_sign
                    
                    if is_lift_connected(signs):
                        r2 = spectral_radius(A)
                        if r2 < current_radius:
                            current_radius = r2
                            best_radius = r2
                            best_signs = signs.copy()
                            improved = True
                        else:
                            signs[i] = old_sign
                            A[u, v] = old_sign
                            A[v, u] = old_sign
                    else:
                        signs[i] = old_sign
                        A[u, v] = old_sign
                        A[v, u] = old_sign
    
    return best_signs
# EVOLVE-BLOCK-END
