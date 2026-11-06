def pebble(dag, fast_memory_budget):
    from collections import deque
    
    if not isinstance(dag, dict):
        return []
    
    # Extract structure
    vertices = dag.get('vertices', dag.get('nodes'))
    edges = dag.get('edges', [])
    
    if vertices is None:
        verts = set()
        for e in edges:
            verts.add(e[0]); verts.add(e[1])
        vertices = sorted(verts) if verts else []
    
    if not edges:
        if 'adj' in dag:
            adj = dag['adj']
            edges = [(u, v) for u in range(len(adj)) for v in adj[u]]
        else:
            # No edges, single vertices
            parents = {v: [] for v in vertices}
            children = {v: [] for v in vertices}
    else:
        parents = {v: [] for v in vertices}
        children = {v: [] for v in vertices}
        for u, v in edges:
            if u in parents and v in children:
                children[u].append(v)
                parents[v].append(u)
            elif v in children:
                children[v].append(u)  # reversed?
                parents[u].append(v)
    
    inputs = dag.get('inputs', dag.get('input_vertices'))
    outputs = dag.get('outputs', dag.get('output_vertices'))
    
    parent_exists = set(p for ps in parents.values() for p in ps)
    child_exists = set(c for cs in children.values() for c in cs)
    
    if inputs is None:
        inputs = [v for v in vertices if v not in parent_exists]
    if outputs is None:
        outputs = [v for v in vertices if v not in child_exists]
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Topo sort
    in_deg = {v: len(parents[v]) for v in vertices}
    q = deque([v for v in vertices if in_deg[v] == 0])
    topo = []
    tmp = in_deg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children[u]:
            tmp[w] -= 1
            if tmp[w] == 0:
                q.append(w)
    if len(topo) < len(vertices):
        topo += [v for v in vertices if v not in topo]
    
    pos = {v: i for i, v in enumerate(topo)}
    last_use = {v: (max(pos[c] for c in children[v]) if children[v] else pos[v]) for v in vertices}
    
    state = {v: ('blue' if v in input_set else 'gone') for v in vertices}
    fast = set()
    ops = []
    n_ops = 0
    
    def add_op(op, v):
        nonlocal n_ops
        if n_ops < 20000:
            ops.append([op, v])
            n_ops += 1
    
    def make_red(v):
        if state[v] == 'red':
            return
        if state[v] == 'blue':
            add_op('load', v)
            state[v] = 'red'
            fast.add(v)
        else:
            for p in parents[v]:
                make_red(p)
            add_op('compute', v)
            state[v] = 'red'
            fast.add(v)
    
    def evict_earliest():
        while len(fast) > fast_memory_budget and n_ops < 20000:
            # Don't evict outputs if possible
            non_out = [f for f in fast if f not in output_set]
            pool = non_out if non_out else list(fast)
            if not pool:
                break
            target = min(pool, key=lambda x: last_use[x])
            add_op('delete', target)
            state[target] = 'gone'
            fast.discard(target)
    
    for v in topo:
        if n_ops >= 20000:
            break
        for p in parents[v]:
            make_red(p)
            evict_earliest()
            if n_ops >= 20000:
                break
        add_op('compute', v)
        state[v] = 'red'
        fast.add(v)
        evict_earliest()
        if v in output_set:
            add_op('store', v)
            state[v] = 'blue'
            fast.discard(v)
    
    for v in output_set:
        if state[v] != 'blue' and n_ops < 20000:
            make_red(v)
            evict_earliest()
            add_op('store', v)
            state[v] = 'blue'
            fast.discard(v)
    
    return ops
