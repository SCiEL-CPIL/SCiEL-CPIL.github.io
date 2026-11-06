def pebble(dag, fast_memory_budget):
    import sys
    sys.setrecursionlimit(10000)
    
    # Determine structure of dag
    # Assume dag is a list of dicts: [{'parents': [...], 'is_output': bool}, ...]
    # Or similar.
    
    if not dag:
        return []
        
    # Try to extract parents and is_output
    # Robust extraction
    N = len(dag)
    parents = []
    is_output = []
    
    for i in range(N):
        node = dag[i]
        if isinstance(node, dict):
            p = node.get('parents', [])
            o = node.get('is_output', False)
        else:
            # Fallback for other structures, e.g. tuple
            try:
                p = node[0]
                o = node[1]
            except:
                p = []
                o = False
        parents.append(p)
        is_output.append(o)
        
    # Build children list
    children = [[] for _ in range(N)]
    for i in range(N):
        for p in parents[i]:
            children[p].append(i)
            
    # child_idx[v] tracks the next child of v to be processed
    child_idx = [0] * N
    
    red_set = set()
    ops = []
    
    def get_next_use(v):
        idx = child_idx[v]
        if idx < len(children[v]):
            return children[v][idx]
        return -1

    for u in range(N):
        # 1. Load parents if not in red_set
        for p in parents[u]:
            if p not in red_set:
                ops.append(('load', p))
                red_set.add(p)
                
        # 2. Compute u
        ops.append(('compute', u))
        red_set.add(u)
        
        # 3. Update child_idx for parents
        for p in parents[u]:
            child_idx[p] += 1
            
        # 4. Eviction
        # We need to ensure len(red_set) <= fast_memory_budget
        while len(red_set) > fast_memory_budget:
            # Find node v in red_set with max next_use
            # If next_use is -1, it's the best candidate (furthest)
            # We want to evict the one with the LARGEST next_use index
            # If multiple have -1, pick any.
            
            max_next = -2
            victim = None
            for v in red_set:
                nu = get_next_use(v)
                # We want to keep nodes with SMALL next_use (needed soon)
                # So we evict the one with LARGEST next_use
                # If nu is -1, it's effectively infinity, so it's a good victim.
                # Let's map -1 to a large number for comparison?
                # No, if nu == -1, it's the best victim.
                # If nu > 0, it's worse than -1.
                
                # Comparison logic:
                # Victim should have the largest "distance to next use".
                # Distance = infinity if -1.
                # Distance = nu if nu >= 0.
                
                if nu == -1:
                    # This is the best victim
                    # But we need to compare with other -1s? Any is fine.
                    # But what if there's a node with nu=100? 100 is worse than -1?
                    # No, -1 means "never used again". So we should evict -1 first.
                    # So -1 is the "largest" distance.
                    
                    # We can just break if we find a -1?
                    # Yes, because -1 is the maximum possible distance.
                    victim = v
                    break
                else:
                    if nu > max_next:
                        max_next = nu
                        victim = v
                        
            if victim is None:
                # Should not happen if red_set is not empty
                break
                
            # Evict victim
            if is_output[victim]:
                ops.append(('store', victim))
            else:
                ops.append(('delete', victim))
            red_set.remove(victim)
            
    # 5. Final cleanup: Ensure all outputs are blue
    # Any node still in red_set must be removed.
    # Outputs must be stored. Non-outputs can be deleted.
    for v in list(red_set):
        if is_output[v]:
            ops.append(('store', v))
        else:
            ops.append(('delete', v))
        red_set.remove(v)
        
    return ops
