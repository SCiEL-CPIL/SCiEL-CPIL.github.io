import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85
    
    try:
        vertices = dag.get_vertices()
        parents = dag.get_parents()
        inputs = dag.get_inputs()
        outputs = dag.get_outputs()
    except:
        if isinstance(dag, dict):
            vertices = dag.get('vertices', list(range(len(dag.get('parents', [])))))
            parents = dag.get('parents', [])
            inputs = dag.get('inputs', [])
            outputs = dag.get('outputs', [])
        else:
            vertices = getattr(dag, 'vertices', list(range(getattr(dag, 'n', 0))))
            parents = getattr(dag, 'parents', [])
            inputs = getattr(dag, 'inputs', [])
            outputs = getattr(dag, 'outputs', [])
    
    n = len(vertices)
    if n == 0:
        return []
    
    children = defaultdict(list)
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Compute last_use: the last vertex index that depends on this vertex
    last_use = [v for v in range(n)]
    for v in range(n - 1, -1, -1):
        for c in children[v]:
            last_use[v] = max(last_use[v], last_use[c])
    
    operations = []
    state = {}
    for v in range(n):
        state[v] = 'blue' if v in input_set else 'gone'
    
    fast_memory = set(inputs)
    
    for v in range(n):
        if time.time() - start_time > DEADLINE:
            break
        
        for p in parents[v]:
            if state[p] == 'red':
                operations.append(['load', p])
                state[p] = 'blue'
                fast_memory.add(p)
        
        if state[v] != 'blue':
            operations.append(['compute', v])
            state[v] = 'blue'
            fast_memory.add(v)
        
        while len(fast_memory) > fast_memory_budget:
            best_victim = None
            best_last_use = -1
            
            for u in fast_memory:
                if u in input_set or u in output_set:
                    continue
                if last_use[u] > best_last_use:
                    best_last_use = last_use[u]
                    best_victim = u
            
            if best_victim is None:
                break
            
            operations.append(['store', best_victim])
            state[best_victim] = 'red'
            fast_memory.discard(best_victim)
    
    for v in outputs:
        if state.get(v, 'gone') == 'red':
            operations.append(['load', v])
            state[v] = 'blue'
            fast_memory.add(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
