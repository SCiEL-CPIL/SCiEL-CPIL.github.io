import random
import itertools

def construct():
    n = 20
    best_edges = []
    best_count = 0
    
    # Run multiple random restarts
    for _ in range(200):
        random.seed() # Or use a specific seed for reproducibility? No, random is fine.
        # Actually, better to use a fixed seed for each restart to be deterministic? 
        # The task doesn't require determinism, but it's good practice. 
        # However, "Maximize" implies finding the global max. 
        # Let's use random restarts.
        
        # List of all possible edges
        edges = list(itertools.combinations(range(n), 2))
        random.shuffle(edges)
        
        # Adjacency list
        adj = [set() for _ in range(n)]
        current_edges = []
        
        for u, v in edges:
            # Check if we can add (u, v)
            # 1. Common neighbors of u and v must be <= 2
            common_uv = adj[u] & adj[v]
            if len(common_uv) > 2:
                continue
                
            # 2. For all w in adj[v] (neighbors of v), 
            #    the new common neighbors of u and w will include v.
            #    So old common neighbors of u and w must be <= 1.
            valid = True
            for w in adj[v]:
                if w == u: continue
                common_uw = adj[u] & adj[w]
                if len(common_uw) > 1:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # 3. For all w in adj[u] (neighbors of u),
            #    the new common neighbors of v and w will include u.
            #    So old common neighbors of v and w must be <= 1.
            for w in adj[u]:
                if w == v: continue
                common_vw = adj[v] & adj[w]
                if len(common_vw) > 1:
                    valid = False
                    break
            
            if valid:
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
    return [[u, v] for u, v in best_edges]
