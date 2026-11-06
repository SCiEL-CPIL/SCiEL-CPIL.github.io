def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    def elapsed():
        return time.time() - start_time
    
    # Extract structure
    if isinstance(dag, dict):
        layers = dag.get('layers', None)
        parents = dag.get('parents', None)
        inputs = dag.get('inputs', None)
        outputs = dag.get('outputs', None)
    else:
        layers = getattr(dag, 'layers', None)
        parents = getattr(dag, 'parents', None)
        inputs = getattr(dag, 'inputs', None)
        outputs = getattr(dag, 'outputs', None)
    
    if layers is None:
        if parents is None:
            return []
        from collections import deque
        all_vertices = set()
        children = {}
        for v, ps in parents.items():
            all_vertices.add(v)
            for p in ps:
                all_vertices.add(p)
                children.setdefault(p, []).append(v)
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        queue = deque([v for v in all_vertices if in_degree[v] == 0])
        layers = []
        current = list(queue)
        while current:
            layers.append(current)
            nxt = []
            for v in current:
                for c in children.get(v, []):
                    in_degree[c] -= 1
                    if in_degree[c] == 0:
                        nxt.append(c)
            current = nxt
        if inputs is None:
            inputs = layers[0] if layers else []
        if outputs is None:
            outputs = layers[-1] if layers else []
    
    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = []
    if parents is None:
        parents = {}
        for i in range(len(layers) - 1):
            for v in layers[i + 1]:
                parents[v] = list(layers[i])
        for v in layers[0] if layers else []:
            parents[v] = []
    
    ops = []
    in_mem = set(inputs)
    
    def do_load(v):
        ops.append(['load', v])
        in_mem.add(v)
    
    def do_store(v):
        ops.append(['store', v])
        in_mem.discard(v)
    
    def do_compute(v):
        ops.append(['compute', v])
        in_mem.add(v)
    
    # Compute for each vertex, the last layer that needs it
    last_needed = {}
    for i, layer in enumerate(layers):
        for v in layer:
            for p in parents.get(v, []):
                if p not in last_needed or i > last_needed[p]:
                    last_needed[p] = i
    
    # Process layers
    for i in range(len(layers)):
        if elapsed() > DEADLINE:
            break
        layer = layers[i]
        
        # Ensure parents are in memory
        if i > 0:
            for v in layer:
                for p in parents.get(v, []):
                    if p not in in_mem:
                        do_load(p)
        
        # Compute layer
        for v in layer:
            if v not in in_mem:
                do_compute(v)
        
        # Free what's no longer needed
        for v in list(in_mem):
            if v in set(outputs):
                continue
            if last_needed.get(v, -1) < i:
                do_store(v)
        
        # Enforce budget
        if len(in_mem) > fast_memory_budget:
            # Free non-outputs with earliest last_needed
            for v in sorted([x for x in in_mem if x not in set(outputs)],
                           key=lambda x: last_needed.get(x, -1)):
                if len(in_mem) <= fast_memory_budget:
                    break
                do_store(v)
    
    # Ensure outputs are blue
    out_set = set(outputs)
    for v in outputs:
        if v not in in_mem:
            if all(p in in_mem for p in parents.get(v, [])):
                do_compute(v)
            else:
                do_load(v)
    
    return ops
