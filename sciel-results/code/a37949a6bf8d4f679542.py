def construct():
    N = 180
    
    # 1. Generate all valid triples (a, b, c)
    # 1/a = 1/b + 1/c => (b-a)(c-a) = a^2
    # Let u = b-a, v = c-a. u*v = a^2.
    # a < b < c => 0 < u < v
    # b <= N => a+u <= N => u <= N-a
    # c <= N => a+v <= N => v <= N-a
    
    triples = []
    for a in range(1, N + 1):
        limit = N - a
        if limit <= 0:
            continue
        a_sq = a * a
        # u must be a divisor of a_sq
        # u < sqrt(a_sq) to ensure u < v
        # Also u <= limit and v <= limit
        for u in range(1, int(a_sq**0.5) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u == v:
                    continue
                if u <= limit and v <= limit:
                    b = a + u
                    c = a + v
                    # b, c are distinct from a and each other
                    triples.append((a, b, c))
    
    if not triples:
        return list(range(1, N + 1))
    
    # 2. Greedy Vertex Cover Removal
    # Precompute adjacency: element -> list of triple indices
    elem_to_triples = [[] for _ in range(N + 1)]
    for i, (a, b, c) in enumerate(triples):
        elem_to_triples[a].append(i)
        elem_to_triples[b].append(i)
        elem_to_triples[c].append(i)
        
    in_set = [True] * (N + 1)
    # degree[i] = number of active triples containing i
    degree = [0] * (N + 1)
    for a, b, c in triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
        
    active_triple_count = len(triples)
    
    # To speed up finding max degree, we can maintain a list of candidates or just scan.
    # Scanning 1..N is O(N). Removals <= N. Total O(N^2) = 32400 ops. Very fast.
    
    while active_triple_count > 0:
        max_deg = -1
        best_elem = -1
        
        for x in range(1, N + 1):
            if in_set[x] and degree[x] > max_deg:
                max_deg = degree[x]
                best_elem = x
                
        if best_elem == -1 or max_deg == 0:
            break
            
        # Remove best_elem
        in_set[best_elem] = False
        
        # Update degrees for all triples containing best_elem
        for t_idx in elem_to_triples[best_elem]:
            a, b, c = triples[t_idx]
            
            # Determine the other two elements
            if best_elem == a:
                o1, o2 = b, c
            elif best_elem == b:
                o1, o2 = a, c
            else:
                o1, o2 = a, b
                
            # If the other two are still in the set, this triple was active.
            # We need to decrement their degrees and the active count.
            if in_set[o1] and in_set[o2]:
                degree[o1] -= 1
                degree[o2] -= 1
                active_triple_count -= 1
                
    return [i for i in range(1, N + 1) if in_set[i]]
