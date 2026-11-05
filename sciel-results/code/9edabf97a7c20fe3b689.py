def pebble(dag, fast_memory_budget):
    if isinstance(dag, dict):
        parents = dag.get('parents', [])
        outputs = set(dag.get('outputs', []))
        n = dag.get('n', len(parents))
        inputs = set(dag.get('inputs', []))
    else:
        parents = getattr(dag, 'parents', [])
        outputs = set(getattr(dag, 'outputs', []))
        n = getattr(dag, 'n', len(parents))
        inputs = set(getattr(dag, 'inputs', []))
    
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Compute last_use: the latest vertex index that transitively depends on v
    last_use = [0] * n
    for v in range(n):
        last_use[v] = v
        for c in children[v]:
            last_use[v] = max(last_use[v], last_use[c])
    
    operations = []
    fast_mem = set(inputs)
    computed = set(inputs)
    
    for v in range(n):
        if v in inputs:
            continue
        if v in computed:
            continue
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in fast_mem:
                if p in computed:
                    operations.append(['load', p])
                    fast_mem.add(p)
        
        # Compute v
        operations.append(['compute', v])
        fast_mem.add(v)
        computed.add(v)
        
        # Evict if over budget
        while len(fast_mem) > fast_memory_budget:
            candidates = [x for x in fast_mem if x not in outputs]
            if not candidates:
                break
            evict_v = max(candidates, key=lambda x: last_use[x])
            operations.append(['store', evict_v])
            fast_mem.discard(evict_v)
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in fast_mem:
            if v in computed:
                operations.append(['load', v])
                fast_mem.add(v)
    
    return operations
