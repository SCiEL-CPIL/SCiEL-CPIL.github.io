def pebble(dag, fast_memory_budget):
    from collections import deque
    
    operations = []
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents_dict = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        
        # Handle case where vertices is a list of dicts
        if vertices and isinstance(vertices[0], dict):
            new_vertices = list(range(len(vertices)))
            new_parents = {}
            new_inputs = set()
            new_outputs = set()
            for i, v in enumerate(vertices):
                new_parents[i] = v.get('parents', [])
                if v.get('is_input', False):
                    new_inputs.add(i)
                if v.get('is_output', False):
                    new_outputs.add(i)
            vertices = new_vertices
            parents_dict = new_parents
            inputs = new_inputs if new_inputs else inputs
            outputs = new_outputs if new_outputs else outputs
    else:
        vertices = list(range(len(dag))) if hasattr(dag, '__len__') else []
        parents_dict = {v: [] for v in vertices}
        inputs = set()
        outputs = set()
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents_dict.get(v, []):
            children[p].append(v)
    
    # Topological sort
    in_deg = {v: len(parents_dict.get(v, [])) for v in vertices}
    q = deque([v for v in vertices if in_deg[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(c)
    if len(topo) < len(vertices):
        topo = list(vertices)
    
    # Pebbling algorithm
    in_mem = set()
    computed = set()
    
    # Load inputs (they start blue)
    for v in sorted(inputs):
        operations.append(['load', v])
        in_mem.add(v)
        computed.add(v)
    
    # Process vertices in topological order
    for v in topo:
        if v in inputs:
            continue
        
        # Ensure all parents are in memory
        for p in parents_dict.get(v, []):
            if p not in in_mem:
                operations.append(['load', p])
                in_mem.add(p)
        
        # Compute vertex
        operations.append(['compute', v])
        in_mem.add(v)
        computed.add(v)
        
        # Evict if over budget
        if len(in_mem) > fast_memory_budget:
            candidates = [x for x in in_mem if x not in outputs and x not in inputs]
            if candidates:
                # Prefer evicting vertices not needed for future computations
                to_evict = None
                for c in candidates:
                    if not any(ch not in computed for ch in children[c]):
                        to_evict = c
                        break
                if to_evict is None:
                    to_evict = candidates[0]
                operations.append(['delete', to_evict])
                in_mem.remove(to_evict)
    
    # Ensure all outputs are in memory
    for v in sorted(outputs):
        if v not in in_mem:
            operations.append(['load', v])
            in_mem.add(v)
    
    return operations
