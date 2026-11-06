def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    
    vertices = list(dag['vertices'])
    parents = dict(dag.get('parents', {}))
    outputs = set(dag.get('outputs', []))
    inputs = set(v for v in vertices if not parents.get(v, []))
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    from collections import deque
    in_deg = {v: len(parents.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_deg[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    
    operations = []
    fast_mem = set()
    
    for v in topo:
        if time.time() - start_time > 85:
            break
        
        for p in parents.get(v, []):
            if p not in fast_mem:
                if p in inputs:
                    operations.append(['load', p])
                    fast_mem.add(p)
                else:
                    for gp in parents.get(p, []):
                        if gp not in fast_mem and gp in inputs:
                            operations.append(['load', gp])
                            fast_mem.add(gp)
                    while len(fast_mem) >= fast_memory_budget:
                        for fm in list(fast_mem):
                            if fm not in inputs and fm != p:
                                operations.append(['delete', fm])
                                fast_mem.discard(fm)
                                break
                        else:
                            break
                    operations.append(['compute', p])
                    fast_mem.add(p)
        
        operations.append(['compute', v])
        fast_mem.add(v)
        
        if v in outputs:
            operations.append(['store', v])
            fast_mem.discard(v)
        
        while len(fast_mem) > fast_memory_budget:
            for fm in list(fast_mem):
                if fm not in inputs:
                    operations.append(['delete', fm])
                    fast_mem.discard(fm)
                    break
            else:
                break
    
    return operations
