import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def _has_c4(adj, n):
        """Check if graph has a C4 using common neighbor count."""
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 1:
                    return True
        return False
    
    def _find_all_c4s(adj, n):
        """Find all pairs (u,v) with >1 common neighbors (C4 indicators)."""
        c4_pairs = []
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if len(common) > 1:
                    c4_pairs.append((u, v, common))
        return c4_pairs
    
    def _try_local_search(n, target_deg, seed, deadline):
        """Initialize with ER random graph, then local search to remove C4s and boost degrees."""
        rng = random.Random(seed)
        
        # Initialize with G(n, p) where p is chosen to give expected degree ~ target_deg
        # For n=47, target_deg=8, p ≈ 8/46 ≈ 0.17
        p = min(0.3, max(0.1, target_deg / max(1, n - 1)))
        
        adj = [set() for _ in range(n)]
        
        # Initialize with random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Local search iterations
        max_iters = 500
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Check current state
            min_deg = min(len(adj[i]) for i in range(n))
            c4_pairs = _find_all_c4s(adj, n)
            
            # If we already have valid solution
            if min_deg >= target_deg and not c4_pairs:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges
            
            # If we have C4s, remove edges to break them
            if c4_pairs:
                # Pick a random C4 pair and remove one edge that helps
                u, v, common = rng.choice(c4_pairs)
                # Remove an edge (u, w) or (v, w) for w in common
                # This will reduce common neighbors
                w = rng.choice(list(common))
                # Decide whether to remove (u,w) or (v,w)
                # Remove the one that causes less degree damage
                if len(adj[u]) <= len(adj[v]):
                    # Remove (u, w)
                    if w in adj[u]:
                        adj[u].discard(w)
                        adj[w].discard(u)
                else:
                    # Remove (v, w)
                    if w in adj[v]:
                        adj[v].discard(w)
                        adj[w].discard(v)
                continue
            
            # No C4s, but need to boost degrees
            # Find vertices below target degree
            under_degree = [i for i in range(n) if len(adj[i]) < target_deg]
            
            if not under_degree:
                # All degrees met, no C4s - should be valid (already checked above)
                break
            
            # Try to add edges to boost under-degree vertices
            added = False
            # Sort under-degree vertices by degree (most under first)
            under_degree.sort(key=lambda i: len(adj[i]))
            
            for u in under_degree:
                if len(adj[u]) >= target_deg:
                    continue
                
                # Try to add an edge from u to some non-neighbor v
                # Prefer v that is also under degree
                candidates = []
                for v in range(n):
                    if v == u or v in adj[u]:
                        continue
                    candidates.append(v)
                
                if not candidates:
                    continue
                
                # Shuffle and try
                rng.shuffle(candidates)
                for v in candidates:
                    # Check if adding (u,v) creates C4
                    # Adding (u,v) creates C4 iff:
                    # 1. |N(u) ∩ N(v)| > 1 (already have >1 common neighbors)
                    # 2. For any z in N(v) \ {u}, |N(u) ∩ N(z)| >= 1 (v would become 2nd common neighbor)
                    # 3. For any z in N(u) \ {v}, |N(v) ∩ N(z)| >= 1
                    
                    if len(adj[u] & adj[v]) > 1:
                        continue
                    
                    safe = True
                    for z in adj[v]:
                        if z == u:
                            continue
                        if len(adj[u] & adj[z]) >= 1:
                            safe = False
                            break
                    
                    if safe:
                        for z in adj[u]:
                            if z == v:
                                continue
                            if len(adj[v] & adj[z]) >= 1:
                                safe = False
                                break
                    
                    if safe:
                        adj[u].add(v)
                        adj[v].add(u)
                        added = True
                        break
            
            # If no progress, add some random perturbation
            if not added:
                # Add a random safe edge
                for _ in range(10):
                    u, v = rng.choice([(a, b) for a in range(n) for b in range(a + 1, n)])
                    if v not in adj[u]:
                        # Check safety
                        if len(adj[u] & adj[v]) > 1:
                            continue
                        safe = True
                        for z in adj[v]:
                            if z == u:
                                continue
                            if len(adj[u] & adj[z]) >= 1:
                                safe = False
                                break
                        if safe:
                            for z in adj[u]:
                                if z == v:
                                    continue
                                if len(adj[v] & adj[z]) >= 1:
                                    safe = False
                                    break
                        if safe:
                            adj[u].add(v)
                            adj[v].add(u)
                            break
        
        # Final check
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= target_deg and not _has_c4(adj, n):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return edges
        
        return None
    
    best_n = 39
    best_edges = []
    
    # Try from highest n down
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
        
        target_deg = max(1, n - 39)
        
        # Try multiple seeds
        max_seeds = 40 if n >= 46 else 80
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            edges = _try_local_search(n, target_deg, seed, deadline)
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                break  # Found valid for this n, try higher n
        
        if best_n > n:
            break
    
    if best_n <= 39:
        return {'n': 39, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
