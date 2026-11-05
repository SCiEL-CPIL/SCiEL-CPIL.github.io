import time
import random

def construct():
    deadline = time.time() + 140
    
    # Try n=47 first
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40]:
        min_deg_req = target_n - 39
        
        # Try multiple random seeds with repair strategy
        for seed in range(2000):
            if time.time() > deadline:
                break
            result = _try_repair(target_n, min_deg_req, seed, deadline)
            if result is not None:
                return result
        
        # Fallback to greedy
        for seed in range(200):
            if time.time() > deadline:
                break
            result = _try_greedy(target_n, min_deg_req, seed, deadline)
            if result is not None:
                return result

    # Fallback
    return {'n': 39, 'edges': []}

def _can_add_edge(adj, u, v, n):
    """Check if adding edge (u,v) creates a C4."""
    # u and v are not currently connected.
    # Check if there is a path u-x-y-v.
    # This exists if there is x in N(u) such that x is connected to some y in N(v).
    nu = adj[u]
    nv = adj[v]
    
    # Iterate over neighbors of u
    x = (nu & -nu).bit_length() - 1
    while nu:
        # If x is connected to any neighbor of v, we have a C4
        if adj[x] & nv:
            return False
        nu &= nu - 1
        if nu:
            x = (nu & -nu).bit_length() - 1
    return True

def _try_repair(n, min_deg_req, seed, deadline):
    rng = random.Random(seed)
    
    # Initial probability
    # We want min degree >= min_deg_req.
    # Let's start with p = 0.2
    p = 0.2
    
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    # Repair loop
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        # 1. Remove C4s
        c4_removed = False
        # Check all pairs for common neighbors >= 2
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common.bit_count() >= 2:
                    # Remove an edge to break the C4
                    # Pick a common neighbor w
                    w = (common & -common).bit_length() - 1
                    # Remove (u, w)
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                    c4_removed = True
                    break
            if c4_removed:
                break
        
        if not c4_removed:
            # No C4s, check degree
            degrees = [adj[u].bit_count() for u in range(n)]
            if min(degrees) >= min_deg_req:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            
            # 2. Increase Minimum Degree
            deg_added = False
            min_deg = min(degrees)
            if min_deg >= min_deg_req:
                break
                
            target_u = degrees.index(min_deg)
            
            # Try to add an edge to target_u
            # Candidates are non-neighbors v such that _can_add_edge(adj, target_u, v, n) is True
            candidates = []
            for v in range(n):
                if v == target_u:
                    continue
                if adj[target_u] & (1 << v):
                    continue
                if _can_add_edge(adj, target_u, v, n):
                    candidates.append(v)
            
            if candidates:
                v = rng.choice(candidates)
                adj[target_u] |= (1 << v)
                adj[v] |= (1 << target_u)
                deg_added = True
            
            if not deg_added:
                break

    return None

def _try_greedy(n, min_deg_req, seed, deadline):
    rng = random.Random(seed)
    adj = [0] * n
    
    for _ in range(2000):
        if time.time() > deadline:
            break
        
        degrees = [adj[u].bit_count() for u in range(n)]
        if min(degrees) >= min_deg_req:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        min_deg = min(degrees)
        if min_deg >= min_deg_req:
            break
            
        target_u = degrees.index(min_deg)
        
        added = False
        non_neighbors = [v for v in range(n) if v != target_u and not (adj[target_u] & (1 << v))]
        rng.shuffle(non_neighbors)
        
        for v in non_neighbors:
            if _can_add_edge(adj, target_u, v, n):
                adj[target_u] |= (1 << v)
                adj[v] |= (1 << target_u)
                added = True
                break
        
        if not added:
            break
            
    return None
