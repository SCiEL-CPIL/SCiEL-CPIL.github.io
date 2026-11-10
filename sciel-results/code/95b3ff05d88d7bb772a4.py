import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 580
    
    p41 = 41
    N41 = set()
    for x in range(1, p41):
        if pow(x, (p41 - 1) // 2, p41) == p41 - 1:
            N41.add(x)
            
    # Build allowed differences
    allowed = np.zeros(m, dtype=bool)
    for d in range(1, m):
        if d % 5 in (2, 3) or d % 41 in N41:
            allowed[d] = True
            
    # Build adjacency sets
    adj = [set() for _ in range(m)]
    for u in range(m):
        for v in range(u + 1, m):
            d = (u - v) % m
            if allowed[d]:
                adj[u].add(v)
                adj[v].add(u)
                
    # Greedy Max Clique
    def greedy_clique():
        clique = []
        candidates = set(range(m))
        while candidates:
            # Pick vertex with max degree in candidates
            v = max(candidates, key=lambda x: len(adj[x] & candidates))
            clique.append(v)
            candidates = candidates & adj[v]
        return clique
        
    best_clique = greedy_clique()
    best_size = len(best_clique)
    
    # Local Search / Improvement
    # Try to swap vertices in the clique with non-clique vertices to increase size
    
    def is_valid_clique(clique):
        clique_set = set(clique)
        for u in clique:
            if not (set(clique) - {u}).issubset(adj[u]):
                return False
        return True
        
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try to add a vertex
        for v in range(m):
            if v in best_clique:
                continue
            # Check if v can be added
            if all(u in adj[v] for u in best_clique):
                best_clique.append(v)
                best_size += 1
                improved = True
                break # Restart loop to check again
                
        if improved:
            continue
            
        # Try to swap
        # Remove one, add one
        for i, u in enumerate(list(best_clique)):
            # Try removing u
            temp_clique = [x for x in best_clique if x != u]
            # Try adding a new vertex v
            for v in range(m):
                if v in temp_clique:
                    continue
                if all(x in adj[v] for x in temp_clique):
                    # Check if this improves size? 
                    # We removed 1, added 1. Size is same.
                    # We need to add MORE than 1?
                    # This simple swap doesn't increase size.
                    # We need to find a set of vertices to replace.
                    pass
                    
    # The simple local search above is weak.
    # Let's use a proper Max Clique search with pruning.
    
    # We will use the Tomita algorithm structure with a simple bound.
    
    def max_clique_search(clique, candidates):
        nonlocal best_clique, best_size
        
        if len(clique) + len(candidates) <= best_size:
            return
            
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique.copy()
            return
            
        # Pick vertex with max degree
        v = max(candidates, key=lambda x: len(adj[x] & candidates))
        
        # Include v
        new_cands = candidates & adj[v]
        clique.append(v)
        max_clique_search(clique, new_cands)
        clique.pop()
        
        # Exclude v
        candidates.discard(v)
        max_clique_search(clique, candidates)
        
    # To make this fast, we rely on the initial greedy clique being good.
    # The search will prune many branches.
    
    # Run search from empty? No, start from the greedy clique?
    # If we start from empty, we might find the same clique.
    # We want to find a LARGER clique.
    
    # We can try to expand the greedy clique.
    # But the search above does that if we start with the greedy clique as the base?
    # No, the search finds the max clique in the induced subgraph of candidates.
    # If we set candidates = V \ best_clique, we are looking for a clique that includes best_clique?
    # No, that's not how it works.
    
    # Let's just run the full search. It's N=205.
    # With a good initial bound, it should be fast.
    
    # To avoid re-finding the same clique, we can just run it.
    
    # One optimization: sort candidates by degree globally?
    
    max_clique_search([], set(range(m)))
    
    return np.array(sorted(best_clique), dtype=np.int64)
