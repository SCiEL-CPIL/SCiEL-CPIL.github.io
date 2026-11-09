def construct():
    import math
    
    N = 256
    # Squares up to 512
    squares = set()
    k = 1
    while k*k <= 512:
        squares.add(k*k)
        k += 1
    
    # First, remove vertices where 2a is a perfect square
    valid_vertices = []
    for a in range(1, N+1):
        if 2*a not in squares:
            valid_vertices.append(a)
    
    # Build conflict graph: edge between a,b (a<b) if a+b is a square
    # Use adjacency sets
    adj = {a: set() for a in valid_vertices}
    for i in range(len(valid_vertices)):
        for j in range(i+1, len(valid_vertices)):
            a = valid_vertices[i]
            b = valid_vertices[j]
            if a + b in squares:
                adj[a].add(b)
                adj[b].add(a)
    
    # Greedy: order by degree ascending, pick if no conflict with already picked
    # Try multiple orderings and take best
    
    def greedy_independent_set(order):
        chosen = set()
        for v in order:
            if v not in chosen:
                # Check if v conflicts with any in chosen
                conflict = False
                for u in chosen:
                    if u in adj[v]:
                        conflict = True
                        break
                if not conflict:
                    chosen.add(v)
        return chosen
    
    # Strategy 1: order by degree ascending
    by_degree_asc = sorted(valid_vertices, key=lambda x: len(adj[x]))
    set1 = greedy_independent_set(by_degree_asc)
    
    # Strategy 2: order by degree descending
    by_degree_desc = sorted(valid_vertices, key=lambda x: -len(adj[x]))
    set2 = greedy_independent_set(by_degree_desc)
    
    # Strategy 3: random orderings
    import random
    random.seed(42)
    best_set = set1 if len(set1) >= len(set2) else set2
    
    for _ in range(200):
        random_order = valid_vertices[:]
        random.shuffle(random_order)
        s = greedy_independent_set(random_order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search: try to add vertices, or swap
    # Try to add any vertex that doesn't conflict
    improved = True
    while improved:
        improved = False
        for v in valid_vertices:
            if v not in best_set:
                conflict = False
                for u in best_set:
                    if u in adj[v]:
                        conflict = True
                        break
                if not conflict:
                    best_set.add(v)
                    improved = True
    
    # Try swaps: remove one, add two
    for v in list(best_set):
        for w in valid_vertices:
            if w in best_set:
                continue
            # Try removing v, adding w
            if w in adj[v]:
                continue
            # Check w doesn't conflict with best_set - {v}
            conflict = False
            for u in best_set:
                if u == v:
                    continue
                if u in adj[w]:
                    conflict = True
                    break
            if not conflict:
                # This swap keeps same size, try double swap
                pass
    
    # Try to add multiple: remove one, add two
    for v in list(best_set):
        remaining = best_set - {v}
        # Find vertices that can be added to remaining
        candidates = []
        for w in valid_vertices:
            if w in best_set:
                continue
            conflict = False
            for u in remaining:
                if u in adj[w]:
                    conflict = True
                    break
            if not conflict:
                candidates.append(w)
        # Try pairs of candidates
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                w1, w2 = candidates[i], candidates[j]
                if w1 in adj[w2]:
                    continue
                # Both can be added
                new_set = remaining | {w1, w2}
                if len(new_set) > len(best_set):
                    best_set = new_set
                    break
            else:
                continue
            break
    
    return list(best_set)
