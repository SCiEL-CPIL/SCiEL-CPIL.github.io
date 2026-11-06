<<<<<<< SEARCH
    # Phase 1: Hybrid Greedy - mix of random and degree-based orderings
    for trial in range(150):
        if time_left() < 10:
            break
            
        adj_try = [0] * n
        ec = 0
        
        # Mix strategies: random, degree-based, or hybrid
        strategy = trial % 3
        if strategy == 0:
            # Pure random
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
        elif strategy == 1:
            # Degree-based: prefer pairs where at least one endpoint has low degree
            # This tends to create more balanced graphs
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
        else:
            # Start with a structured base (e.g., complete bipartite-ish) then random
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time_left() < 5:
                break
            
            if can_add(u, v):
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
                ec += 1
        
        if ec > best_edges:
            best_edges = ec
            best_adj = adj_try[:]
=======
    # Phase 1: Hybrid Greedy - mix of random and degree-based orderings
    for trial in range(150):
        if time_left() < 10:
            break
            
        adj_try = [0] * n
        ec = 0
        
        # Mix strategies: random, degree-based, or hybrid
        strategy = trial % 3
        if strategy == 0:
            # Pure random
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
        elif strategy == 1:
            # Degree-based: sort by sum of degrees (ascending) to add low-degree edges first
            # Re-sort periodically as degrees change
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
            # Initial sort by random to break ties
            for i in range(len(shuffled)):
                if i % 20 == 0:
                    # Re-sort remaining by current degree sum
                    remaining = shuffled[i:]
                    remaining.sort(key=lambda p: adj_try[p[0]].bit_count() + adj_try[p[1]].bit_count())
                    shuffled[i:] = remaining
        else:
            # Structured: try to build a regular-ish graph
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time_left() < 5:
                break
            
            if can_add(u, v):
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
                ec += 1
        
        if ec > best_edges:
            best_edges = ec
            best_adj = adj_try[:]
>>>>>>> REPLACE

