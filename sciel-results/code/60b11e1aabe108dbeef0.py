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
    
    def _try_greedy(n, target_deg, seed, deadline):
        """Try to build C4-free graph on n vertices with min_degree >= target_deg."""
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        
        # Precompute all edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        def is_safe(u, v):
            """Check if adding edge (u,v) keeps graph C4-free."""
            # Check (u,v) common neighbors
            if len(adj[u] & adj[v]) > 1:
                return False
            
            # Check pairs (u, z) for z in N(v) where z != u
            for z in adj[v]:
                if z == u:
                    continue
                if len(adj[u] & adj[z]) >= 1:
                    return False
            
            # Check pairs (v, z) for z in N(u) where z != v
            for z in adj[u]:
                if z == v:
                    continue
                if len(adj[v] & adj[z]) >= 1:
                    return False
            
            return True
        
        # Shuffle edge order for diversity based on seed
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        # Multi-pass greedy: in each pass, find the best edge to add
        # Use deficit-based key to prioritize most under-degree vertices
        for pass_num in range(n * 4):
            if time.time() > deadline:
                break
            
            best_edge = None
            best_key = None
            
            for u, v in shuffled_edges:
                if v in adj[u]:
                    continue
                
                deg_u = len(adj[u])
                deg_v = len(adj[v])
                
                # Only consider if it helps at least one under-degree vertex
                if deg_u >= target_deg and deg_v >= target_deg:
                    continue
                
                if not is_safe(u, v):
                    continue
                
                # Deficit-based key: prioritize edges that reduce the max deficit
                # deficit = target_deg - deg (how far below target)
                deficit_u = target_deg - deg_u
                deficit_v = target_deg - deg_v
                # We want to minimize the max deficit after adding
                # Adding (u,v) reduces deficit_u by 1 and deficit_v by 1
                new_deficit_u = deficit_u - 1
                new_deficit_v = deficit_v - 1
                # Key: minimize the resulting max deficit, then total deficit
                key = (max(new_deficit_u, new_deficit_v), new_deficit_u + new_deficit_v)
                
                if best_key is None or key < best_key:
                    best_key = key
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            
            # Early exit if target reached
            if all(len(adj[i]) >= target_deg for i in range(n)):
                break
        
        # Final verification
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= target_deg:
            if not _has_c4(adj, n):
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
        
        # Try multiple seeds - more attempts for harder cases
        max_seeds = 80 if n >= 46 else 120
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            edges = _try_greedy(n, target_deg, seed, deadline)
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                break  # Found valid for this n, move to next (higher) n
        
        if best_n >= n:
            if best_n > n:
                break
            continue
    
    if best_n <= 39:
        return {'n': 39, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
