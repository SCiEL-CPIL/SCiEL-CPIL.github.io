import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    deadline = time.time() + 85.0
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Identify edge types: top cycle (0..m-1), bottom cycle (m..2m-1), matching (i, i+m)
    top_edges = []    # edges in top cycle
    bot_edges = []    # edges in bottom cycle
    match_edges = []  # matching edges
    edge_type = []    # 'T', 'B', 'M'
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(i)
            edge_type.append('T')
        elif u >= m and v >= m:
            bot_edges.append(i)
            edge_type.append('B')
        else:
            match_edges.append(i)
            edge_type.append('M')
    
    # For prism graph: top cycle has m edges, bottom cycle has m edges, matching has m edges
    # Connectivity: product of signs around top cycle * product around bottom cycle must be -1
    # Actually, the lift is connected iff the signed graph is not balanced.
    # For a cycle, balanced iff product of signs = +1.
    # The prism has two independent cycles (top and bottom) plus the matching.
    # The lift is connected iff NOT (top cycle product = +1 AND bottom cycle product = +1)
    # Wait, let me think more carefully.
    
    # The 2-lift: vertex (v,0) and (v,1). Edge (u,v) with sign s connects:
    #   +1: (u,0)-(v,0) and (u,1)-(v,1)
    #   -1: (u,0)-(v,1) and (u,1)-(v,0)
    
    # The lift is connected iff there's no vertex labeling f: V -> {0,1} such that
    # s(u,v) = (-1)^(f(u)+f(v)) for all edges.
    
    # For prism graphs, we can check this efficiently.
    
    def is_connected(signs):
        """Check if the 2-lift is connected using BFS on the signed graph."""
        f = [-1] * n
        f[0] = 0
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
        
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                flip = 1 if s == -1 else 0
                required = f[u] ^ flip
                if f[v] == -1:
                    f[v] = required
                    queue.append(v)
                else:
                    if f[v] != required:
                        return True  # Contradiction -> not a coboundary -> connected
        return False  # No contradiction -> coboundary exists -> disconnected
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            val = float(signs[i])
            S[u, v] = val
            S[v, u] = val
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))
    
    # Precompute edge arrays for fast matrix construction
    u_arr = np.array([e[0] for e in edges], dtype=np.int32)
    v_arr = np.array([e[1] for e in edges], dtype=np.int32)
    
    def fast_radius(signs):
        """Fast spectral radius computation."""
        S = np.zeros((n, n), dtype=np.float64)
        vals = signs.astype(np.float64)
        S[u_arr, v_arr] = vals
        S[v_arr, u_arr] = vals
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))
    
    best_signs = None
    best_radius = np.inf
    
    rng = np.random.default_rng(3)
    
    # For small m, do exhaustive search
    if num_edges <= 16:
        # Exhaustive search
        max_combo = 1 << num_edges
        for k in range(max_combo):
            if time.time() > deadline - 5.0:
                break
            signs = np.where((k >> np.arange(num_edges)) & 1, 1, -1).astype(np.int8)
            if not is_connected(signs):
                continue
            r = fast_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
        
        if best_signs is not None:
            return best_signs
    
    # For larger m, use multi-start simulated annealing + hill climbing
    
    def hill_climb(signs, current_radius):
        """Local search: flip individual edges to improve."""
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(num_edges):
                old_sign = signs[i]
                signs[i] = -old_sign
                if not is_connected(signs):
                    signs[i] = old_sign
                    continue
                new_r = fast_radius(signs)
                if new_r < current_radius:
                    current_radius = new_r
                    improved = True
                    if new_r < best_radius:
                        nonlocal best_radius, best_signs
                        best_radius = new_r
                        best_signs = signs.copy()
                else:
                    signs[i] = old_sign
        return current_radius
    
    # Multi-start SA
    num_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 5.0 and num_restarts < max_restarts:
        num_restarts += 1
        
        # Random initial state
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        
        # Ensure connectivity
        attempts = 0
        while not is_connected(signs) and attempts < 100:
            idx = rng.integers(0, num_edges)
            signs[idx] *= -1
            attempts += 1
        if not is_connected(signs):
            continue
        
        current_radius = fast_radius(signs)
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()
        
        # Simulated annealing
        T = 1.0
        T_end = 0.005
        cooling = 0.9995
        
        while T > T_end and time.time() < deadline - 5.0:
            for _ in range(30):
                if time.time() > deadline - 5.0:
                    break
                
                idx = rng.integers(0, num_edges)
                old_sign = signs[idx]
                signs[idx] = -old_sign
                
                if not is_connected(signs):
                    signs[idx] = old_sign
                    continue
                
                new_r = fast_radius(signs)
                delta = new_r - current_radius
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_radius = new_r
                    if new_r < best_radius:
                        best_radius = new_r
                        best_signs = signs.copy()
                else:
                    signs[idx] = old_sign
            
            T *= cooling
        
        # Hill climb to polish
        current_radius = hill_climb(signs.copy(), current_radius)
    
    # Final hill climb on best
    if best_signs is not None:
        final_radius = hill_climb(best_signs.copy(), best_radius)
        if final_radius < best_radius:
            best_radius = final_radius
    
    if best_signs is None:
        # Fallback
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not is_connected(best_signs):
            best_signs[0] = -1
            if not is_connected(best_signs):
                best_signs[1] = -1
    
    return best_signs
