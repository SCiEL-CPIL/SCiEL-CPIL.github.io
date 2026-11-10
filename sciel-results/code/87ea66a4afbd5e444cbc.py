def _exact_score(points):
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear3(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    
    if not triples:
        return n

    best = 0
    
    # Precompute for each vertex, the list of triples it's in
    # And for each pair, if they form a triple with v
    
    def search(selected_mask, available_mask, current_size):
        nonlocal best
        if available_mask == 0:
            if current_size > best:
                best = current_size
            return
            
        if current_size + bin(available_mask).count('1') <= best:
            return
            
        # Pick lowest bit
        v = (available_mask & -available_mask).bit_length() - 1
        
        # Exclude v
        search(selected_mask, available_mask & ~(1 << v), current_size)
        
        # Include v
        # Check validity
        valid = True
        for t in triples:
            if v in t:
                others = [x for x in t if x != v]
                if (selected_mask >> others[0]) & 1 and (selected_mask >> others[1]) & 1:
                    valid = False
                    break
                    
        if valid:
            search(selected_mask | (1 << v), available_mask & ~(1 << v), current_size + 1)

    search(0, (1 << n) - 1, 0)
    return best
