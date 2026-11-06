def pebble(dag, fast_memory_budget):
    # ---- introspect dag structure ----
    def get(dag, names):
        # try attribute then dict key
        for n in names:
            try:
                v = getattr(dag, n)
                if v is not None:
                    return v
            except Exception:
                pass
            try:
                if isinstance(dag, dict) and n in dag:
                    return dag[n]
            except Exception:
                pass
        return None

    parents_map = None
    vertices = None
    outputs = None
    inputs = None

    # try explicit parents
    parents_map = get(dag, ['parents', 'parent', 'preds', 'predecessors', 'in_edges', 'dependencies'])
    outputs = get(dag, ['outputs', 'output_vertices', 'output', 'sinks', 'targets'])
    inputs = get(dag, ['inputs', 'input_vertices', 'input', 'sources'])

    # if dag is a dict mapping vertex->parents
    if parents_map is None and isinstance(dag, dict):
        # check if values are lists of vertices
        sample = list(dag.values())[:1]
        if sample and isinstance(sample[0], (list, tuple, set)):
            parents_map = {k: list(v) for k, v in dag.items()}

    if parents_map is None:
        # maybe dag is a list of (parent, child) edges
        if isinstance(dag, (list, tuple)):
            edges = dag
            parents_map = {}
            vertices = set()
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    p, c = e
                    parents_map.setdefault(c, []).append(p)
                    vertices.add(p); vertices.add(c)
            if vertices:
                parents_map = {v: parents_map.get(v, []) for v in vertices}

    if parents_map is None:
        raise ValueError("cannot interpret dag")

    # normalize parents_map
    parents_map = {v: list(ps) for v, ps in parents_map.items()}
    if vertices is None:
        vertices = set(parents_map.keys())
        for ps in parents_map.values():
            for p in ps:
                vertices.add(p)
    vertices = set(vertices)
    # ensure every vertex in parents_map
    for v in vertices:
        parents_map.setdefault(v, [])

    # children
    children = {v: [] for v in vertices}
    for v, ps in parents_map.items():
        for p in ps:
            children[p].append(v)

    if outputs is None:
        outputs = [v for v in vertices if not children[v]]
    outputs = set(outputs)
    if inputs is None:
        inputs = [v for v in vertices if not parents_map[v]]
    inputs = set(inputs)

    # topological order (Kahn)
    indeg = {v: len(parents_map[v]) for v in vertices}
    from collections import deque
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    indeg2 = dict(indeg)
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children[v]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    if len(topo) != len(vertices):
        # cycle or missing; append remaining
        for v in vertices:
            if v not in topo:
                topo.append(v)

    budget = int(fast_memory_budget)
    if budget < 1:
        budget = 1

    blue = set(inputs)
    ops = []
    remaining_children = {v: len(children[v]) for v in vertices}
    topo_index = {v: i for i, v in enumerate(topo)}

    def pick_victim():
        # prefer delete candidates: not output, remaining_children==0
        del_cands = [v for v in blue if v not in outputs and remaining_children[v] == 0]
        if del_cands:
            # evict the one with earliest topo index (frees oldest)
            del_cands.sort(key=lambda v: topo_index[v])
            return del_cands[0], 'delete'
        # else store: pick farthest next use
        def next_use(v):
            # earliest child in topo that still needs v
            best = None
            for c in children[v]:
                if remaining_children[v] > 0:
                    idx = topo_index[c]
                    if best is None or idx < best:
                        best = idx
            return best
        store_cands = [v for v in blue if v not in outputs]
        if store_cands:
            # pick the one with farthest next use (or no use)
            def score(v):
                nu = next_use(v)
                return (nu if nu is not None else float('inf'))
            store_cands.sort(key=lambda v: -score(v))
            return store_cands[0], 'store'
        # fallback: any non-output, else output
        any_cands = [v for v in blue]
        if any_cands:
            any_cands.sort(key=lambda v: (v in outputs, topo_index[v]))
            return any_cands[0], 'store'
        return None, None

    for v in topo:
        # ensure parents blue
        for p in parents_map[v]:
            if p not in blue:
                ops.append(['load', p])
                blue.add(p)
        ops.append(['compute', v])
        if v not in blue:
            blue.add(v)
        for p in parents_map[v]:
            remaining_children[p] -= 1
        # evict
        while len(blue) > budget:
            victim, op = pick_victim()
            if victim is None:
                break
            ops.append([op, victim])
            blue.remove(victim)
        if len(ops) > 20000:
            break

    # ensure outputs blue
    for o in outputs:
        if o not in blue:
            ops.append(['load', o])
            blue.add(o)

    # trim to 20000
    return ops[:20000]
