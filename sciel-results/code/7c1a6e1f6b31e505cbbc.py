def pebble(dag, fast_memory_budget):
    """
    Minimize communication (transfers) for DAG pebbling.
    
    dag: dict with 'vertices', 'parents', 'outputs' OR a parent map {v: [p1, p2]}
         - If 'vertices' exists, uses standard dict format.
         - If 'parents' exists, uses {parents: {...}, outputs: [...]}
         - Otherwise, assumes dag is the parent map {v: [p1, p2]}
    fast_memory_budget: int, number of fast memory slots available
    
    Returns: list of [operation, vertex]
    """
    import time
    start_time = time.time()
    
    # Robust Parsing
    vertices = []
    parents = {}
    outputs = set()
    
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = dag['vertices']
            parents = dag.get('parents', {})
            outputs = set(dag.get('outputs', []))
        elif 'parents' in dag:
            parents = dag['parents']
            vertices = list(parents.keys())
            outputs = set(dag.get('outputs', []))
        else:
            # Assume dag is the parent map
            parents = dag
            vertices = list(dag.keys())
            # Infer outputs: vertices with no children
            children_map_temp = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    if p in children_map_temp:
                        children_map_temp[p].append(v)
            outputs = {v for v in vertices if not children_map_temp[v]}
    else:
        # Fallback
        raise ValueError("Unsupported DAG format")

    # Build children map
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children_map:
                children_map[p].append(v)

    # Topological Sort
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
        
    if len(topo_order) != len(vertices):
        topo_order = vertices

    # Pebbling Strategy
    operations = []
    in_mem = set()
    computed = set()
    
    def remaining_children_count(v):
        count = 0
        for c in children_map.get(v, []):
            if c not in computed:
                count += 1
        return count

    def evict_if_needed():
        nonlocal in_mem
        if len(in_mem) < fast_memory_budget:
            return
        # Prefer evicting non-outputs with 0 remaining children
        candidates = [x for x in in_mem if x not in outputs and remaining_children_count(x) == 0]
        if not candidates:
            # Prefer evicting non-outputs
            candidates = [x for x in in_mem if x not in outputs]
        if not candidates:
            # Evict any
            candidates = list(in_mem)
        
        if candidates:
            evict = candidates[0]
            operations.append(['store', evict])
            in_mem.remove(evict)

    for v in topo_order:
        # Deadline check
        if time.time() - start_time > 50:
            break

        is_input = (len(parents.get(v, [])) == 0)
        
        if is_input:
            if v not in in_mem:
                evict_if_needed()
                operations.append(['load', v])
                in_mem.add(v)
            # Inputs don't need compute
        else:
            # Non-input
            # Ensure parents are in memory
            for p in parents.get(v, []):
                if p not in in_mem:
                    evict_if_needed()
                    operations.append(['load', p])
                    in_mem.add(p)
            
            # Compute v
            operations.append(['compute', v])
            in_mem.add(v)
            computed.add(v)

    # Final cleanup: Store all remaining in memory
    for v in list(in_mem):
        operations.append(['store', v])
        in_mem.remove(v)

    return operations
