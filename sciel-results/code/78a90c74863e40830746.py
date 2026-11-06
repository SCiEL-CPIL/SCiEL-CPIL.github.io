import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, 10s margin for safety
    
    # Parse DAG
    if isinstance(dag, dict):
        parents_dict = {k: list(v) for k, v in dag.items()}
        n = max(parents_dict.keys()) + 1 if parents_dict else 0
    elif isinstance(dag, (list, tuple)):
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)):
            parents_dict = {i: list(dag[i]) for i in range(len(dag))}
            n = len(dag)
        else:
            parents_dict = {}
            for edge in dag:
                if len(edge) == 2:
                    p, c = edge
                    if c not in parents_dict:
                        parents_dict[c] = []
                    parents_dict[c].append(p)
            n = max(parents_dict.keys()) + 1 if parents_dict else 0
    else:
        return []
    
    if n == 0:
        return []
    
    for i in range(n):
        if i not in parents_dict:
            parents_dict[i] = []
    
    parents = [parents_dict[i] for i in range(n)]
    
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    inputs = [i for i in range(n) if len(parents[i]) == 0]
    outputs = [i for i in range(n) if len(children[i]) == 0]
    
    layers = [0] * n
    for i in range(n):
        if parents[i]:
            layers[i] = 1 + max(layers[p] for p in parents[i])
    
    max_layer = max(layers) if n > 0 else 0
    layer_vertices = [[] for _ in range(max_layer + 1)]
    for i in range(n):
        layer_vertices[layers[i]].append(i)
    
    num_children = [len(children[i]) for i in range(n)]
    
    operations = []
    memory = set()
    slow_memory = set()
    
    for v in inputs:
        operations.append(['load', v])
        memory.add(v)
    
    for layer in range(1, max_layer + 1):
        for v in layer_vertices[layer]:
            for p in parents[v]:
                if p not in memory:
                    if len(memory) >= fast_memory_budget:
                        evictable = [x for x in memory if x not in parents[v]]
                        if evictable:
                            evict_candidate = min(evictable, key=lambda x: num_children[x])
                            operations.append(['store', evict_candidate])
                            memory.discard(evict_candidate)
                            slow_memory.add(evict_candidate)
                    operations.append(['load', p])
                    memory.add(p)
                    slow_memory.discard(p)
            
            if len(memory) >= fast_memory_budget:
                evictable = [x for x in memory if x not in parents[v] and x != v]
                if evictable:
                    evict_candidate = min(evictable, key=lambda x: num_children[x])
                    operations.append(['store', evict_candidate])
                    memory.discard(evict_candidate)
                    slow_memory.add(evict_candidate)
            
            operations.append(['compute', v])
            memory.add(v)
    
    for v in outputs:
        if v in memory:
            operations.append(['store', v])
            memory.discard(v)
            slow_memory.add(v)
    
    return operations
