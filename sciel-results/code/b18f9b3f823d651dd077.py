def pebble(dag, fast_memory_budget):
    from collections import deque
    
    def get_attr(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
    
    vertices = get_attr(dag, 'vertices', None)
    parents = get_attr(dag, 'parents', None)
    outputs = get_attr(dag, 'outputs', None)
    
    if vertices is None:
        if isinstance(dag, list):
            vertices = dag
        else:
            raise ValueError("Cannot extract vertices")
    
    if parents is None:
        parents = {}
    if outputs is None:
        outputs = []
    
    # Ensure parents is a dict
    if not isinstance(parents, dict):
        # Maybe it's a list of lists
        parents = {v: parents[i] for i, v in enumerate(vertices)}
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    inputs = [v for v in vertices if not parents.get(v, [])]
    
    # Topological order
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    in_degree_copy = dict(in_degree)
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree_copy[c] -= 1
            if in_degree_copy[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        topo_order = list(vertices)
    
    topo_index = {v: i for i, v in enumerate(topo_order)}
    outputs_set = set(outputs)
    inputs_set = set(inputs)
    
    last_use = {}
    for v in vertices:
        lu = topo_index[v]
        if v in outputs_set:
            lu = len(topo_order) - 1
        for c in children[v]:
            if topo_index[c] > lu:
                lu = topo_index[c]
        last_use[v] = lu
    
    fast = set()
    slow = set()
    ops = []
    transfers = 0
    computed = set()
    
    # Inputs start in fast memory (blue)
    for v in inputs:
        fast.add(v)
        computed.add(v)
    
    def evict_if_needed(protected=None):
        nonlocal fast, slow, ops, transfers
        if len(fast) < fast_memory_budget:
            return
        if protected is None:
            protected = set()
        candidates = [x for x in fast if x not in outputs_set and x not in protected]
        if not candidates:
            return False
        victim = max(candidates, key=lambda x: last_use[x])
        ops.append(('store', victim))
        transfers += 1
        fast.remove(victim)
        slow.add(victim)
        return True
    
    def ensure_in_fast(v):
        nonlocal fast, slow, ops, transfers
        if v in fast:
            return
        if v in slow:
            if not evict_if_needed():
                # Can't evict, but need space
                # Try to evict even outputs? No.
                # This is a problem
                pass
            ops.append(('load', v))
            transfers += 1
            fast.add(v)
            slow.remove(v)
        else:
            # v not in fast or slow, recompute
            # For now, raise error
            raise ValueError(f"Vertex {v} not available")
    
    def compute_vertex(v):
        nonlocal fast, slow, ops, transfers, computed
        parent_set = set(parents.get(v, []))
        for p in parent_set:
            ensure_in_fast(p)
        if not evict_if_needed(protected=parent_set):
            # Can't evict, but need space for v
            # This means budget is too small
            # Try to proceed anyway (might exceed budget)
            pass
        ops.append(('compute', v))
        fast.add(v)
        computed.add(v)
    
    for v in topo_order:
        if v in inputs_set:
            continue
        compute_vertex(v)
    
    # Ensure outputs in fast
    for v in outputs:
        ensure_in_fast(v)
    
    return ops
