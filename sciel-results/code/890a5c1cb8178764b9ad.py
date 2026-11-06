import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try to find a valid graph for n=35
    n = 35
    # Partition into 3 parts
    # Sizes: 12, 12, 11
    parts = [list(range(12)), list(range(12, 24)), list(range(24, 35))]
    
    # Initialize adjacency matrix (list of sets)
    adj = [set() for _ in range(n)]
    
    # Add all edges between different parts
    for i in range(n):
        for j in range(i+1, n):
            # Check if i and j are in different parts
            part_i = 0 if i < 12 else (1 if i < 24 else 2)
            part_j = 0 if j < 12 else (1 if j < 24 else 2)
            if part_i != part_j:
                adj[i].add(j)
                adj[j].add(i)
                
    # Candidate edges: within parts
    candidates = []
    for p in parts:
        for i in range(len(p)):
            for j in range(i+1, len(p)):
                candidates.append((p[i], p[j]))
                
    random.shuffle(candidates)
    
    # Add candidates if they don't create K4
    for u, v in candidates:
        if time.time() > deadline:
            break
        # Check if adding (u,v) creates a K4
        # Common neighbors
        cn = adj[u].intersection(adj[v])
        # Check if any edge exists in cn
        creates_k4 = False
        for w in cn:
            if w in adj[v] and w in adj[u]: # w is in cn, so this is true
                # Check if w is connected to any other x in cn
                # Actually, if there is an edge (w, x) in the subgraph induced by cn
                pass
        # Faster check:
        # For each w in cn, if w has a neighbor in cn, then K4 is formed?
        # No, if w and x are in cn and w~x, then {u,v,w,x} is K4.
        # So we just need to check if the subgraph induced by cn has any edge.
        k4_found = False
        cn_list = list(cn)
        for idx, w in enumerate(cn_list):
            for x in cn_list[idx+1:]:
                if x in adj[w]:
                    k4_found = True
                    break
            if k4_found:
                break
        
        if not k4_found:
            adj[u].add(v)
            adj[v].add(u)
            
    # Check for Independent Set of size 6
    def has_indep_6(adj, n):
        # Backtracking search for independent set of size 6
        # Use bitmasks or sets
        # Since n=35, we can use a simple recursive search with pruning
        # Order vertices by degree (ascending) to prune faster?
        # Or just pick vertices one by one.
        
        def backtrack(remaining, size):
            if size == 6:
                return True
            if not remaining:
                return False
            # Pruning: if len(remaining) < size, return False
            if len(remaining) < size:
                return False
                
            # Pick a vertex v
            # Heuristic: pick vertex with smallest degree in the induced subgraph of remaining?
            # Or just pick the first one.
            v = next(iter(remaining))
            
            # Option 1: Include v
            # New remaining: vertices in remaining that are not neighbors of v and not v
            new_remaining = set()
            for u in remaining:
                if u != v and u not in adj[v]:
                    new_remaining.add(u)
            if backtrack(new_remaining, size - 1):
                return True
                
            # Option 2: Exclude v
            remaining.remove(v)
            if backtrack(remaining, size):
                return True
            remaining.add(v)
            
            return False
            
        return backtrack(set(range(n)), 6)

    if has_indep_6(adj, n):
        # Failed to break all I6
        # Try a different random seed or strategy?
        # For now, return the graph if it's valid, else try n=34?
        # Actually, if it fails, we can try to run a local search to fix it.
        # But for simplicity, if it fails, we might return a smaller valid graph.
        # Let's try n=34 if 35 fails?
        # Or just return the best valid n found.
        pass
    else:
        # Valid graph on 35 vertices
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
        
    # If 35 failed, try a simpler construction for smaller n
    # E.g. Paley 17 is known to work.
    # We can try to construct n=20, 25, etc.
    # For now, fallback to Paley 17 to ensure score > 1.0
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
