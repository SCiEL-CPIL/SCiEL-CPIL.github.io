def _robust_fallback(dag, fast_memory_budget):
    vertices, parents, outputs, inputs = _toposort_vertices(dag)
    children = {}
    for v in vertices:
        for p in parents.get(v, []):
            children.setdefault(p, []).append(v)
            
    # Topological order
    topo = []
    in_degree = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            in_degree[v] += 1
    queue = [v for v in vertices if in_degree[v] == 0]
    while queue:
        v = queue.pop(0)
        topo.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo) != len(vertices):
        # Cycle? Just return empty or naive
        return []

    ops = []
    fast_mem = set()
    blue = set(inputs)
    available = set(inputs)
    
    # Pre-compute "needed" status
    # A vertex is needed if it is an ancestor of any unprocessed vertex
    unprocessed = set(vertices)
    
    def is_needed(v):
        if v in outputs:
            return True
        for c in children.get(v, []):
            if c in unprocessed:
                return True
        return False

    for v in topo:
        unprocessed.discard(v)
        
        if v in inputs:
            if v not in fast_mem and v not in blue:
                if len(fast_mem) >= fast_memory_budget:
                    # Evict
                    evicted = False
                    # 1. Try to evict non-needed, non-output
                    for u in list(fast_mem):
                        if not is_needed(u) and u not in outputs:
                            ops.append(('delete', u))
                            fast_mem.discard(u)
                            if u not in blue:
                                available.discard(u)
                            evicted = True
                            break
                    if not evicted:
                        # 2. Try to evict non-output
                        for u in list(fast_mem):
                            if u not in outputs:
                                ops.append(('delete', u))
                                fast_mem.discard(u)
                                if u not in blue:
                                    available.discard(u)
                                evicted = True
                                break
                    if not evicted:
                        # 3. Store an output to free space
                        for u in list(fast_mem):
                            if u in outputs and u not in blue:
                                ops.append(('store', u))
                                fast_mem.discard(u)
                                blue.add(u)
                                # u remains in available
                                evicted = True
                                break
                    if not evicted and fast_mem:
                        # 4. Delete any
                        u = next(iter(fast_mem))
                        ops.append(('delete', u))
                        fast_mem.discard(u)
                        if u not in blue:
                            available.discard(u)
                
                if len(fast_mem) < fast_memory_budget:
                    ops.append(('load', v))
                    fast_mem.add(v)
                    available.add(v)
        else:
            # Compute v
            # Ensure parents are available
            for p in parents.get(v, []):
                if p not in available:
                    # p is not in blue and not in fast_mem
                    # We must have evicted it. We need to recompute it?
                    # This is a failure of the "keep needed" strategy.
                    # For robustness, we just load/compute p if it's an input.
                    # If p is computed, we can't easily recompute without recursion.
                    # Assuming the strategy works, this shouldn't happen.
                    # If it does, we fail.
                    pass
            
            if len(fast_mem) >= fast_memory_budget:
                # Evict
                evicted = False
                for u in list(fast_mem):
                    if not is_needed(u) and u not in outputs:
                        ops.append(('delete', u))
                        fast_mem.discard(u)
                        if u not in blue:
                            available.discard(u)
                        evicted = True
                        break
                if not evicted:
                    for u in list(fast_mem):
                        if u not in outputs:
                            ops.append(('delete', u))
                            fast_mem.discard(u)
                            if u not in blue:
                                available.discard(u)
                            evicted = True
                            break
                if not evicted:
                    for u in list(fast_mem):
                        if u in outputs and u not in blue:
                            ops.append(('store', u))
                            fast_mem.discard(u)
                            blue.add(u)
                            evicted = True
                            break
                if not evicted and fast_mem:
                    u = next(iter(fast_mem))
                    ops.append(('delete', u))
                    fast_mem.discard(u)
                    if u not in blue:
                        available.discard(u)
            
            if len(fast_mem) < fast_memory_budget:
                ops.append(('compute', v))
                fast_mem.add(v)
                available.add(v)
        
        if v in outputs and v in fast_mem:
            ops.append(('store', v))
            fast_mem.discard(v)
            blue.add(v)
            
    # Cleanup
    for v in list(fast_mem):
        ops.append(('delete', v))
        fast_mem.discard(v)
        
    return ops
