import random

def construct():
    # Target n=46, min degree 7
    # Theoretical max min degree is floor((1 + sqrt(4n-3))/2) which is ~7 for n=46
    # We try n=46 first.
    
    for n in [46, 45, 44, 43, 42, 41, 40, 39]:
        min_deg_req = n - 39
        if min_deg_req < 0:
            min_deg_req = 0
            
        # Try multiple random initializations and repairs
        for attempt in range(200):
            adj = [set() for _ in range(n)]
            deg = [0] * n
            
            # Phase 1: Random Greedy Edge Addition
            # Generate all pairs
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                # Check if adding (u, v) creates a C4
                # C4 exists if there is w in N(u) such that w in N(v)
                # i.e., N(u) & N(v) is not empty
                if not adj[u].intersection(adj[v]):
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
            
            # Phase 2: Repair to satisfy min degree
            # If min degree is already satisfied, we can skip
            if min(deg) < min_deg_req:
                improved = True
                while improved:
                    improved = False
                    # Find vertex with minimum degree
                    min_idx = min(range(n), key=lambda i: deg[i])
                    if deg[min_idx] >= min_deg_req:
                        break
                    
                    u = min_idx
                    # Find candidates v
                    candidates = []
                    for v in range(n):
                        if v == u or v in adj[u]:
                            continue
                        # Check C4 constraint
                        if not adj[u].intersection(adj[v]):
                            candidates.append(v)
                    
                    if candidates:
                        # Pick a random candidate to avoid bias
                        v = random.choice(candidates)
                        adj[u].add(v)
                        adj[v].add(u)
                        deg[u] += 1
                        deg[v] += 1
                        improved = True
                        # Continue loop to check if min degree is now satisfied or needs more repair
                    else:
                        # No candidates for u, try next lowest? 
                        # Or just fail this attempt
                        # For now, break if we can't improve the current min vertex
                        # But maybe another vertex is the bottleneck? 
                        # Since we picked min_idx, if we can't add to it, 
                        # adding to others won't help it.
                        # So this attempt is likely stuck.
                        break

            if min(deg) >= min_deg_req:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
        
    # Fallback
    return {'n': 39, 'edges': []}
