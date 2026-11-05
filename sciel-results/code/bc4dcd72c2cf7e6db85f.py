def _greedy_schedule(topo_order, parents_map, children_map, use_indices, index_of, inputs, outputs, relevant, budget, start_time, deadline):
    import bisect
    import time
    
    blue = set()
    red = set()
    ops = []
    transfers = 0
    
    # Initialize: inputs that are relevant are blue
    for v in inputs:
        if v in relevant:
            blue.add(v)
    
    # Process each vertex in topo_order
    for i, v in enumerate(topo_order):
        if time.time() - start_time > deadline:
            break
        
        # If v is an input, it's already blue (or red if stored)
        # Ensure parents are blue
        for p in parents_map.get(v, []):
            if p in relevant and p not in blue:
                # p is red, load it
                if p in red:
                    ops.append(['load', p])
                    blue.add(p)
                    red.remove(p)
                    transfers += 1
                else:
                    # p is gone, should not happen
                    pass
        
        # Compute v
        ops.append(['compute', v])
        blue.add(v)
        
        # If over budget, evict
        while len(blue) > budget:
            # Find the blue vertex with the furthest next use
            best_v = None
            best_next = -1
            for b in blue:
                nu = _next_use_of(b, i, use_indices, index_of, outputs)
                if nu > best_next:
                    best_next = nu
                    best_v = b
            if best_v is None:
                break
            ops.append(['store', best_v])
            blue.remove(best_v)
            red.add(best_v)
            transfers += 1
    
    # Ensure outputs are blue
    for o in outputs:
        if o in relevant and o not in blue:
            if o in red:
                ops.append(['load', o])
                blue.add(o)
                red.remove(o)
                transfers += 1
    
    return ops
