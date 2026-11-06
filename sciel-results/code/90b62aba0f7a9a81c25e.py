def pebble(dag, fast_memory_budget):
    if isinstance(dag, dict):
        parents = dag['parents']
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
        if not inputs:
            inputs = [i for i in range(len(parents)) if len(parents[i]) == 0]
        if not outputs:
            is_parent = [False] * len(parents)
            for p_list in parents:
                for p in p_list:
                    is_parent[p] = True
            outputs = [i for i in range(len(parents)) if not is_parent[i]]
    else:
        parents = dag
        n = len(parents)
        inputs = [i for i in range(n) if len(parents[i]) == 0]
        is_parent = [False] * n
        for p_list in parents:
            for p in p_list:
                is_parent[p] = True
        outputs = [i for i in range(n) if not is_parent[i]]
    
    n = len(parents)
    if n == 0:
        return []
    
    state = [0] * n
    blue_count = 0
    operations = []
    
    def compute(v):
        nonlocal blue_count
        if state[v] == 2:
            return
        for p in parents[v]:
            compute(p)
        operations.append(['compute', v])
        state[v] = 2
        blue_count += 1
    
    for v in range(n):
        compute(v)
        if v not in outputs and blue_count > fast_memory_budget:
            # Delete oldest non-output
            for u in range(v):
                if state[u] == 2 and u not in outputs:
                    operations.append(['delete', u])
                    state[u] = 0
                    blue_count -= 1
                    break
    
    for v in outputs:
        compute(v)
    
    return operations
