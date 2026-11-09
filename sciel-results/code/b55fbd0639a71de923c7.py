import random

def construct():
    # Points are 0..15
    n = 16
    points = list(range(n))
    
    # Generate all triples
    all_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triples.append((i, j, k))
                
    # Identify star triples (containing 0)
    star_triples = set(t for t in all_triples if 0 in t)
    non_star_triples = [t for t in all_triples if 0 not in t]
    
    # Precompute for each non-star triple, the sets of other 7 edges for each K(2,2,2) containing it
    # We store these as a list of sets of tuples for fast intersection/lookup
    # Actually, we just need to check if the other 7 are in the current set S.
    # So for a triple t, we need a list of 7-tuples (the other edges).
    
    k_conflicts = {t: [] for t in non_star_triples}
    
    for t in non_star_triples:
        a, b, c = t
        # The other 3 vertices u, v, w must be distinct from a,b,c
        # Pairs are {a,u}, {b,v}, {c,w}
        # The 8 edges are:
        # (a,b,c)
        # (a,b,w), (a,v,c), (u,b,c)
        # (a,v,w), (u,b,w), (u,v,c)
        # (u,v,w)
        
        others = [x for x in points if x not in t]
        for u in others:
            for v in others:
                if v == u: continue
                for w in others:
                    if w == u or w == v: continue
                    # Construct the 7 other edges
                    # Note: edges are sets of 3, order doesn't matter for storage in set, 
                    # but we store as sorted tuples.
                    e1 = tuple(sorted((a, b, w)))
                    e2 = tuple(sorted((a, v, c)))
                    e3 = tuple(sorted((u, b, c)))
                    e4 = tuple(sorted((a, v, w)))
                    e5 = tuple(sorted((u, b, w)))
                    e6 = tuple(sorted((u, v, c)))
                    e7 = tuple(sorted((u, v, w)))
                    
                    k_conflicts[t].append((e1, e2, e3, e4, e5, e6, e7))
                    
    best_size = 0
    best_set = None
    
    # Run greedy multiple times with different random orders
    for _ in range(20): # 20 restarts
        random.shuffle(non_star_triples)
        current_set = set(star_triples)
        
        for t in non_star_triples:
            # Check if adding t creates a K(2,2,2)
            is_safe = True
            for conflict in k_conflicts[t]:
                # Check if all 7 other edges are in current_set
                if all(e in current_set for e in conflict):
                    is_safe = False
                    break
            
            if is_safe:
                current_set.add(t)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    # Convert back to list of lists
    return [list(t) for t in best_set]
