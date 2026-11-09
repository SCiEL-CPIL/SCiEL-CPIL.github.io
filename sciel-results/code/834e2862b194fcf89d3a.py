def pebble(dag, fast_memory_budget):
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
        layers = dag.get('layers', None)
        edges = dag.get('edges', None)
    else:
        vertices = list(dag.keys()) if hasattr(dag, 'keys') else []
        parents = dict(dag) if hasattr(dag, 'keys') else {}
        inputs = []
        outputs = []
        layers = None
        edges = None
    
    if not parents and edges:
        parents = {v: [] for v in vertices}
        for p, c in edges:
            parents.setdefault(c, []).append(p)
    
    for v in vertices:
        if v not in parents:
            parents[v] = []
    
    if not inputs:
        inputs = [v for v in vertices if not parents.get(v, [])]
    if not outputs:
        children_map = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                children_map[p].append(v)
        outputs = [v for v in vertices if not children_map.get(v, [])]
    
    if layers is None:
        depth = {}
        def get_depth(v):
            if v in depth:
                return depth[v]
            if not parents.get(v, []):
                depth[v] = 0
                return 0
            max_d = max(get_depth(p) for p in parents[v])
            depth[v] = max_d + 1
            return max_d + 1
        
        for v in vertices:
            get_depth(v)
        
        max_depth = max(depth.values()) if depth else 0
        layers = [[] for _ in range(max_depth + 1)]
        for v in vertices:
            layers[depth[v]].append(v)
    
    operations = []
    in_fast_memory = set()
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Process layer by layer
    for layer_idx, layer in enumerate(layers):
        if not layer:
            continue
        
        to_compute = [v for v in layer if v not in input_set]
        
        # Load needed parents not in fast memory
        needed_parents = set()
        for v in to_compute:
            for p in parents.get(v, []):
                needed_parents.add(p)
        
        for p in needed_parents:
            if p not in in_fast_memory:
                operations.append(['load', p])
                in_fast_memory.add(p)
        
        # Compute vertices in this layer
        for v in to_compute:
            operations.append(['compute', v])
            in_fast_memory.add(v)
        
        # Delete vertices not needed in future and not outputs
        # Determine which vertices are needed in future
        future_needed = set()
        for future_layer in layers[layer_idx + 1:]:
            for v in future_layer:
                for p in parents.get(v, []):
                    future_needed.add(p)
        
        for v in layer:
            if v not in future_needed and v not in output_set:
                operations.append(['delete', v])
                in_fast_memory.discard(v)
    
    # Ensure all outputs are in fast memory
    for v in outputs:
        if v not in in_fast_memory:
            operations.append(['load', v])
            in_fast_memory.add(v)
    
    return operations
