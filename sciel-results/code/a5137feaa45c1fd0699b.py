import time
import numpy as np

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    E = len(edges)
    
    def build_signed_matrix(signs):
        M = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            M[u, v] = s
            M[v, u] = s
        return M
    
    def spectral_radius(signs):
        M = build_signed_matrix(signs)
        ev = np.linalg.eigvalsh(M)
        return np.max(np.abs(ev))
    
    def is_connected_lift(signs):
        adj = [[] for _ in range(2 * n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)
        
        visited = [False] * (2 * n)
        visited[0] = True
        queue = [0]
        while queue:
            cur = queue.pop()
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        return all(visited)
    
    rng = np.random.default_rng(123)
    
    best_signs = None
    best_radius = float('inf')
    
    while time.time() - start_time < DEADLINE:
        signs = rng.choice([-1, 1], size=E)
        
        if not is_connected_lift(signs):
            for attempt in range(500):
                idx = rng.integers(0, E)
                signs[idx] = -signs[idx]
                if is_connected_lift(signs):
                    break
            if not is_connected_lift(signs):
                continue
        
        radius = spectral_radius(signs)
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
        
        current_signs = signs.copy()
        current_radius = radius
        T = 0.05
        
        while time.time() - start_time < DEADLINE:
            idx = rng.integers(0, E)
            old_sign = current_signs[idx]
            current_signs[idx] = -old_sign
            
            new_radius = spectral_radius(current_signs)
            new_connected = is_connected_lift(current_signs)
            
            if new_connected:
                if new_radius < current_radius:
                    current_radius = new_radius
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_signs = current_signs.copy()
                elif rng.random() < np.exp(-(new_radius - current_radius) / T):
                    current_radius = new_radius
            else:
                current_signs[idx] = old_sign
            
            T *= 0.999
            if T < 1e-5:
                break
    
    if best_signs is None:
        best_signs = np.ones(E, dtype=int)
    
    return best_signs
