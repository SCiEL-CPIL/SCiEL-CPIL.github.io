def pebble(dag, fast_memory_budget):
    from collections import deque, OrderedDict
    
    # Parse DAG
    parents = dag['parents']  # dict: vertex -> list of parent vertices
    outputs = set(dag['outputs'])
    
    # All vertices
    all_vertices = set(parents.keys())
    for p_list in parents.values():
        for p in p_list:
            all_vertices.add(p)
    
    # Inputs: vertices with no parents
    inputs = set(v for v in all_vertices if not parents.get(v, []))
    
    # Children: vertex -> list of children
    children = {v: [] for v in all_vertices}
    for v, p_list in parents.items():
        for p in p_list:
            children[p].append(v)
    
    # Topological order (Kahn's algorithm)
    in_deg = {v: len(parents.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_deg[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    
    # State
    fast = set(inputs)  # vertices in fast memory
    slow = set()  # vertices in slow memory (computed and stored)
    computed = set(inputs)  # vertices that have been computed
    ops = []
    
    # LRU order: OrderedDict for efficient LRU
    lru = OrderedDict()
    for v in inputs:
        lru[v] = True
    
    def load(v):
        if v in fast:
            return
        if v in slow:
            ops.append(['load', v])
            fast.add(v)
            slow.discard(v)
            lru[v] = True
            lru.move_to_end(v)
        else:
            # v is not available; this is an error
            raise Exception(f"Cannot load {v}")
    
    def store(v):
        if v not in fast:
            return
        ops.append(['store', v])
        fast.discard(v)
        slow.add(v)
        if v in lru:
            del lru[v]
    
    # Process in topological order
    for v in topo_order:
        if v in computed:
            continue
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast:
                if p in slow:
                    load(p)
                else:
                    raise Exception(f"Parent {p} of {v} not available")
        # Compute v
        ops.append(['compute', v])
        fast.add(v)
        computed.add(v)
        lru[v] = True
        lru.move_to_end(v)
        
        # Evict if over budget
        while len(fast) > fast_memory_budget:
            # Find the least recently used vertex that is not an output
            evict = None
            for candidate in lru:
                if candidate not in outputs:
                    evict = candidate
                    break
            if evict is None:
                # All are outputs; evict the least recently used
                evict = next(iter(lru))
            store(evict)
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in fast:
            if o in slow:
                load(o)
            else:
                raise Exception(f"Output {o} not available")
    
    return ops
