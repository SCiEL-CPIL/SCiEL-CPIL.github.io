def construct():
    triples = P['edge_order']
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Precompute bitmask for each triple
    triple_masks = []
    for t in triples:
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks.append(mask)
        
    # Precompute for each vertex, which triples (indices) contain it
    # Indices are sorted in increasing order
    vertex_to_triples = [[] for _ in range(n)]
    for i, t in enumerate(triples):
        for v in t:
            vertex_to_triples[v].append(i)

    def find_matching_in_prefix(k):
        """
        Find 6 disjoint triples from indices [0, k) covering all vertices.
        Returns list of indices or None.
        """
        best_result = [None]
        
        def backtrack(used_mask, chosen):
            if len(chosen) == 6:
                if used_mask == full_mask:
                    best_result[0] = list(chosen)
                    return True
                return False
            
            # Find unused vertex with fewest available options
            min_count = float('inf')
            best_vertex = -1
            best_options = []
            
            for v in range(n):
                if used_mask & (1 << v):
                    continue
                    
                options = []
                for i in vertex_to_triples[v]:
                    if i >= k:
                        break
                    if used_mask & triple_masks[i]:
                        continue
                    options.append(i)
                    
                if len(options) < min_count:
                    min_count = len(options)
                    best_vertex = v
                    best_options = options
                    if min_count == 0:
                        return False
            
            if min_count == 0:
                return False
                
            for i in best_options:
                chosen.append(i)
                if backtrack(used_mask | triple_masks[i], chosen):
                    return True
                chosen.pop()
                
            return False
            
        backtrack(0, [])
        return best_result[0]

    # Binary search for the minimal prefix size
    lo, hi = 6, len(triples)
    
    # Check if solution exists in full list
    if not find_matching_in_prefix(hi):
        # Fallback
        return [list(range(i, i+3)) for i in range(0, n, 3)]
        
    while lo < hi:
        mid = (lo + hi) // 2
        if find_matching_in_prefix(mid):
            hi = mid
        else:
            lo = mid + 1
            
    result_indices = find_matching_in_prefix(lo)
    
    if result_indices is None:
        return [list(range(i, i+3)) for i in range(0, n, 3)]
        
    return [list(triples[i]) for i in result_indices]
