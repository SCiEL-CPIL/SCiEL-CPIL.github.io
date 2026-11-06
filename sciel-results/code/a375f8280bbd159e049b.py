def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0

    # Parse DAG structure
    if isinstance(dag, dict):
        parents = dict(dag.get('parents', {}))
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        vertices = set(parents.keys()) | set(inputs) | set(outputs)
        for ps in parents.values():
            vertices.update(ps)
    else:
        raise ValueError(f"Unknown dag format: {type(dag)}")

    # Build children adjacency
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)

    # Assign layers: 0=input, 1=middle, 2=output
    layer = {}
    for v in vertices:
        if v in inputs:
            layer[v] = 0
        elif v in outputs:
            layer[v] = 2
        else:
            layer[v] = 1

    blue = set(inputs)
    ops = []

    # Process middle-layer vertices in topological order (sorted by fan-in for locality)
    middle = [v for v in vertices if layer[v] == 1]
    middle.sort(key=lambda v: len(parents.get(v, [])))

    for v in middle:
        if time.time() - start_time > DEADLINE:
            break
        ps = parents.get(v, [])
        for p in ps:
            if p not in blue:
                if len(blue) >= fast_memory_budget:
                    # Evict: prefer inputs no longer needed by unprocessed middle vertices
                    ev = None
                    for b in blue:
                        if b in inputs:
                            remaining = [c for c in children.get(b, [])
                                         if layer.get(c, 0) == 1 and c != v and c not in blue]
                            if not remaining:
                                ev = b
                                break
                    if ev is None:
                        for b in blue:
                            if b not in outputs:
                                ev = b
                                break
                    if ev is not None:
                        ops.append(['store', ev])
                        blue.discard(ev)
                ops.append(['load', p])
                blue.add(p)
        ops.append(['compute', v])
        blue.add(v)

    # Process output vertices
    for v in outputs:
        if time.time() - start_time > DEADLINE:
            break
        ps = parents.get(v, [])
        for p in ps:
            if p not in blue:
                if len(blue) >= fast_memory_budget:
                    ev = None
                    for b in blue:
                        if b not in outputs:
                            ev = b
                            break
                    if ev is not None:
                        ops.append(['store', ev])
                        blue.discard(ev)
                ops.append(['load', p])
                blue.add(p)
        ops.append(['compute', v])
        blue.add(v)

    return ops
