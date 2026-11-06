def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    # Robust Parsing
    vertices = []
    parents = []
    outputs = []
    
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = dag['vertices']
            # Map vertex names to indices if they are strings
            v_to_idx = {v: i for i, v in enumerate(vertices)}
            
            # Get parents
            raw_parents = dag.get('parents', dag.get('edges', {}))
            if isinstance(raw_parents, dict):
                # keys are children, values are lists of parents
                # Need to map to indices
                parents = []
                for v in vertices:
                    p_list = raw_parents.get(v, [])
                    parents.append([v_to_idx[p] for p in p_list if p in v_to_idx])
            elif isinstance(raw_parents, list):
                # Assume list of lists, already indexed or matching vertices order
                parents = [list(p) for p in raw_parents]
            else:
                parents = [[] for _ in vertices]
                
            outputs = dag.get('outputs', dag.get('sinks', []))
            if isinstance(outputs, list) and outputs and isinstance(outputs[0], str):
                outputs = [v_to_idx[o] for o in outputs if o in v_to_idx]
            elif not outputs:
                # Find sinks
                children = [[] for _ in range(len(vertices))]
                for i, p_list in enumerate(parents):
                    for p in p_list:
                        children[p].append(i)
                outputs = [i for i in range(len(vertices)) if not children[i]]
                
    elif isinstance(dag, list):
        n = len(dag)
        vertices = list(range(n))
        parents = []
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                p_list = v.get('parents', v.get('inputs', []))
                # If parents are strings, need mapping, but here we assume indices or simple list
                # If dag is list of dicts, keys might be 'id', 'parents'
                # Let's assume parents are already indices or we need to map if they are strings
                # For simplicity, assume they are indices or we can map if we have a global map
                # Since we don't have a global map in this branch, assume indices.
                parents.append(p_list)
                if v.get('is_output', False):
                    outputs.append(i)
            else:
                parents.append([])
        
        if not outputs:
             # Find sinks
             children = [[] for _ in range(n)]
             for i, p_list in enumerate(parents):
                 for p in p_list:
                     if 0 <= p < n:
                         children[p].append(i)
             outputs = [i for i in range(n) if not children[i]]
    else:
        # Fallback
        return []

    n = len(vertices)
    if n == 0:
        return []
        
    # Ensure parents are valid indices
    valid_parents = []
    for i in range(n):
        vp = []
        for p in parents[i]:
            if isinstance(p, int) and 0 <= p < n:
                vp.append(p)
        valid_parents.append(vp)
    parents = valid_parents
    
    # Build children
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
            
    # Topological Sort
    in_degree = [len(parents[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != n:
        # Cycle or invalid graph, fallback to range
        topo_order = list(range(n))
        
    # Greedy Pebbling
    schedule = []
    in_memory = set()
    
    # Precompute remaining uses
    remaining_uses = [0] * n
    for i in range(n):
        for p in parents[i]:
            remaining_uses[p] += 1
            
    is_output = [False] * n
    for o in outputs:
        if 0 <= o < n:
            is_output[o] = True
            
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
            
        # Load parents
        for p in parents[v]:
            if p not in in_memory:
                schedule.append(('load', p))
                in_memory.add(p)
                
        # Compute
        schedule.append(('compute', v))
        in_memory.add(v)
        
        # Update remaining uses for parents
        for p in parents[v]:
            remaining_uses[p] -= 1
            
        # Evict if over budget
        # We want to keep v if it's an output? No, outputs must be stored (blue) at end.
        # But we can store them immediately to free space.
        
        # If v is an output, store it immediately to free slot and satisfy "finish blue"
        if is_output[v]:
            if v in in_memory:
                schedule.append(('store', v))
                in_memory.discard(v)
        
        while len(in_memory) > fast_memory_budget:
            # Find best candidate to evict
            best_evict = None
            best_score = float('inf')
            
            for u in in_memory:
                if u == v and not is_output[v]:
                    # Don't evict the just computed non-output vertex immediately
                    # (Though we might have to if budget is 0, but budget >= 1 usually)
                    continue
                    
                # Score: lower is better to evict
                # If remaining_uses is 0, it's a sink in the remaining graph.
                # We can delete it (free) or store it (cost 1).
                # Delete is better.
                if remaining_uses[u] == 0:
                    score = -1 # Priority to delete
                else:
                    # Score by remaining uses. Lower uses -> evict sooner.
                    score = remaining_uses[u]
                    
                if score < best_score:
                    best_score = score
                    best_evict = u
                    
            if best_evict is None:
                # Only v is in memory and it's protected?
                # Force evict v?
                if v in in_memory:
                    best_evict = v
                else:
                    break
                    
            if best_evict == v and is_output[v]:
                # Already handled above, but if it's still there, store it
                schedule.append(('store', v))
                in_memory.discard(v)
            elif remaining_uses[best_evict] == 0:
                schedule.append(('delete', best_evict))
                in_memory.discard(best_evict)
            else:
                schedule.append(('store', best_evict))
                in_memory.discard(best_evict)
                
    # Post-pass: Ensure all outputs are stored (blue)
    # If an output was not stored during the loop (e.g. due to deadline break),
    # and it is currently in memory, store it.
    # If it is not in memory, it was either stored or deleted.
    # If it was deleted, that's bad. But our logic tries to avoid deleting outputs
    # by storing them immediately.
    
    # Check for any outputs still in memory
    for o in outputs:
        if o in in_memory:
            schedule.append(('store', o))
            in_memory.discard(o)
            
    # Clean up remaining memory
    for v in list(in_memory):
        schedule.append(('delete', v))
        in_memory.discard(v)
        
    if len(schedule) > 20000:
        schedule = schedule[:20000]
        
    return schedule
