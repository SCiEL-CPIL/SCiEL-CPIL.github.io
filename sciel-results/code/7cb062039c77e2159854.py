def pebble(dag, fast_memory_budget):
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    topo_order = list(vertices)
    
    operations = []
    fast_mem = set()  # currently in fast memory (red)
    blue = set(inputs)  # currently blue
    
    for v in topo_order:
        if v in inputs:
            continue
        
        # Check parents are blue
        for p in parents.get(v, []):
            if p not in blue:
                if p in fast_mem:
                    operations.append(['store', p])
                    fast_mem.discard(p)
                    blue.add(p)
                else:
                    operations.append(['load', p])
                    fast_mem.add(p)
                    operations.append(['store', p])
                    fast_mem.discard(p)
                    blue.add(p)
        
        operations.append(['compute', v])
        fast_mem.add(v)
        
        # Decide fate of v
        future_children = children[v]
        needs_future = len(future_children) > 0 or v in outputs
        
        if v in outputs:
            # Must be blue at end
            if len(fast_mem) > fast_memory_budget:
                for x in list(fast_mem):
                    if x != v and x not in outputs and x not in inputs:
                        operations.append(['delete', x])
                        fast_mem.discard(x)
                        break
            operations.append(['store', v])
            fast_mem.discard(v)
            blue.add(v)
        elif needs_future and len(fast_mem) <= fast_memory_budget:
            # Keep in fast memory for now
            pass
        elif needs_future:
            # Fast memory full, evict
            for x in list(fast_mem):
                if x != v and x not in outputs:
                    operations.append(['delete', x])
                    fast_mem.discard(x)
                    break
            operations.append(['store', v])
            fast_mem.discard(v)
            blue.add(v)
        else:
            operations.append(['delete', v])
            fast_mem.discard(v)
    
    for v in outputs:
        if v not in blue:
            if v in fast_mem:
                operations.append(['store', v])
                fast_mem.discard(v)
                blue.add(v)
    
    for v in list(fast_mem):
        operations.append(['delete', v])
    
    return operations
