import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    # Build adjacency structure
    adj = [[] for _ in range(2*m)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    def build_signed_adj(signs):
        A = np.zeros((2*m, 2*m), dtype=float)
        for idx, (u, v) in enumerate(edges):
            s = float(signs[idx])
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(A):
        vals = np.linalg.eigvalsh(A)
        return np.max(np.abs(vals))
    
    def is_valid_lift(signs):
        """Check if the 2-lift is connected and 3-regular."""
        # 3-regular is guaranteed by construction (prism is 3-regular, lift preserves degree)
        # Check connectivity of the 2-lift
        # Vertices in lift: (v, 0) and (v, 1) for v in 0..2m-1
        # Total lift vertices: 4m
        n_lift = 4 * m
        lift_adj = [[] for _ in range(n_lift)]
        
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                # Connect same sides
                lift_adj[u*2 + 0].append(v*2 + 0)
                lift_adj[v*2 + 0].append(u*2 + 0)
                lift_adj[u*2 + 1].append(v*2 + 1)
                lift_adj[v*2 + 1].append(u*2 + 1)
            else:
                # Connect opposite sides
                lift_adj[u*2 + 0].append(v*2 + 1)
                lift_adj[v*2 + 1].append(u*2 + 0)
                lift_adj[u*2 + 1].append(v*2 + 0)
                lift_adj[v*2 + 0].append(u*2 + 1)
        
        # BFS/DFS to check connectivity
        visited = [False] * n_lift
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            node = stack.pop()
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    stack.append(neighbor)
        
        return count == n_lift
    
    # Initial heuristic: alternate signs on cycles, +1 on matching
    # We need to figure out which edges are in top cycle, bottom cycle, matching
    # Top cycle: edges between vertices in 0..m-1
    # Bottom cycle: edges between vertices in m..2m-1
    # Matching: edges between i and i+m
    
    signs = np.ones(n_edges, dtype=int)
    
    # Identify edge types
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(idx)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    # For top cycle, alternate signs
    for i, idx in enumerate(top_cycle_edges):
        signs[idx] = 1 if i % 2 == 0 else -1
    
    # For bottom cycle, alternate signs
    for i, idx in enumerate(bottom_cycle_edges):
        signs[idx] = 1 if i % 2 == 0 else -1
    
    # Matching: start with +1
    for idx in matching_edges:
        signs[idx] = 1
    
    # Check validity, if not valid, try flipping some matching edges
    if not is_valid_lift(signs):
        # Try flipping one matching edge to make it connected
        for idx in matching_edges:
            trial = signs.copy()
            trial[idx] = -trial[idx]
            if is_valid_lift(trial):
                signs = trial
                break
    
    A = build_signed_adj(signs)
    best_radius = spectral_radius(A)
    best_signs = signs.copy()
    
    # Local search: flip individual signs
    while time.time() - start_time < time_limit:
        improved = False
        for idx in range(n_edges):
            if time.time() - start_time > time_limit - 5:
                break
            
            trial = best_signs.copy()
            trial[idx] = -trial[idx]
            
            if not is_valid_lift(trial):
                continue
            
            A_trial = build_signed_adj(trial)
            r_trial = spectral_radius(A_trial)
            
            if r_trial < best_radius - 1e-10:
                best_radius = r_trial
                best_signs = trial
                improved = True
        
        if not improved:
            # Try pair flips if stuck
            pair_improved = False
            for i in range(n_edges):
                if time.time() - start_time > time_limit - 5:
                    break
                for j in range(i+1, n_edges):
                    if time.time() - start_time > time_limit - 5:
                        break
                    
                    trial = best_signs.copy()
                    trial[i] = -trial[i]
                    trial[j] = -trial[j]
                    
                    if not is_valid_lift(trial):
                        continue
                    
                    A_trial = build_signed_adj(trial)
                    r_trial = spectral_radius(A_trial)
                    
                    if r_trial < best_radius - 1e-10:
                        best_radius = r_trial
                        best_signs = trial
                        pair_improved = True
                        break
                if pair_improved:
                    break
            
            if not pair_improved:
                # Try random restarts with perturbation
                trial = best_signs.copy()
                # Flip 2-3 random edges
                n_flips = np.random.randint(2, 4)
                flip_indices = np.random.choice(n_edges, size=n_flips, replace=False)
                for idx in flip_indices:
                    trial[idx] = -trial[idx]
                
                if is_valid_lift(trial):
                    A_trial = build_signed_adj(trial)
                    r_trial = spectral_radius(A_trial)
                    if r_trial < best_radius - 1e-10:
                        best_radius = r_trial
                        best_signs = trial
                        pair_improved = True
            
            if not pair_improved:
                break
    
    return best_signs
