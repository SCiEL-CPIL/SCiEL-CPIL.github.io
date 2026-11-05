def pebble(dag, fast_memory_budget):
    if isinstance(dag, dict):
        n = dag.get('n', len(dag.get('parents', [])))
        parents = dag.get('parents', [])
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        topo_order = dag.get('topo_order', list(range(n)))
    else:
        n = len(dag)
        parents = dag
        inputs = set()
        outputs = set()
        topo_order = list(range(n))
    
    operations = []
    in_fast = set()
    
    future_need = [0] * n
    for v in range(n):
        for p in parents[v]:
            future_need[p] += 1
    
    for v in topo_order:
        for p in parents[v]:
            if p not in in_fast:
                operations.append(['load', p])
                in_fast.add(p)
        
        operations.append(['compute', v])
        in_fast.add(v)
        
        for p in parents[v]:
            future_need[p] -= 1
        
        while len(in_fast) > fast_memory_budget:
            candidates = [x for x in in_fast if x != v]
            if not candidates:
                break
            
            # Pick vertex to evict: min future_need, prefer non-output
            best = min(candidates, key=lambda x: (future_need[x], x in outputs, x))
            
            if best in outputs and future_need[best] == 0:
                operations.append(['store', best])
            else:
                operations.append(['delete', best])
            in_fast.remove(best)
    
    for v in outputs:
        if v in in_fast:
            operations.append(['store', v])
            in_fast.remove(v)
    
    for v in list(in_fast):
        operations.append(['delete', v])
        in_fast.remove(v)
    
    return operations
