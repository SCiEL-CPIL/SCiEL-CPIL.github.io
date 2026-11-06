def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    if isinstance(dag, dict):
        parents_map = {v: list(p) for v, p in dag.items()}
        vertices = list(parents_map.keys())
    elif isinstance(dag, list):
        parents_map = {}
        if len(dag) > 0 and isinstance(dag[0], dict):
            for item in dag:
                v = item.get('vertex', item.get('id', item.get('node')))
                p = item.get('parents', item.get('preds', item.get('inputs', [])))
                parents_map[v] = list(p)
            vertices = list(parents_map.keys())
        else:
            vertices = []
            for item in dag:
                if isinstance(item, (tuple, list)):
                    v = item[0]
                    p = list(item[1])
                    parents_map[v] = p
                    vertices.append(v)
                else:
                    vertices.append(item)
                    parents_map[item] = []
    else:
        raise ValueError(f"Unknown dag format")
    
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents_map[v]:
            if p in children_map:
                children_map[p].append(v)
    
    outputs = [v for v in vertices if not children_map.get(v, [])]
    if not outputs:
        outputs = vertices
    
    from collections import deque
    in_deg = {v: len(parents_map[v]) for v in vertices}
    q = deque([v for v in vertices if in_deg[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children_map.get(v, []):
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(c)
    
    pebbled = set()
    ops = []
    unprocessed = set(topo)
    
    def remaining_uses(v):
        return sum(1 for c in children_map.get(v, []) if c in unprocessed)
    
    for v in topo:
        if time.time() - start_time > DEADLINE:
            break
        unprocessed.discard(v)
        
        if v not in pebbled:
            for p in parents_map[v]:
                if p not in pebbled:
                    ops.append(['load', p])
                    pebbled.add(p)
                    while len(pebbled) > fast_memory_budget:
                        best = None
                        best_score = float('inf')
                        for x in pebbled:
                            if x == v or x in parents_map[v]:
                                continue
                            s = remaining_uses(x)
                            if s < best_score:
                                best_score = s
                                best = x
                        if best is not None:
                            ops.append(['delete', best])
                            pebbled.discard(best)
                        else:
                            break
            
            ops.append(['compute', v])
            pebbled.add(v)
            
            while len(pebbled) > fast_memory_budget:
                best = None
                best_score = float('inf')
                for x in pebbled:
                    if x == v:
                        continue
                    s = remaining_uses(x)
                    if s < best_score:
                        best_score = s
                        best = x
                if best is not None:
                    ops.append(['delete', best])
                    pebbled.discard(best)
                else:
                    break
        
        if remaining_uses(v) == 0 and v not in outputs:
            if len(pebbled) >= fast_memory_budget:
                ops.append(['delete', v])
                pebbled.discard(v)
    
    for v in outputs:
        if v not in pebbled:
            for p in parents_map[v]:
                if p not in pebbled:
                    ops.append(['load', p])
                    pebbled.add(p)
                    while len(pebbled) > fast_memory_budget:
                        best = None
                        best_score = float('inf')
                        for x in pebbled:
                            if x == v or x in parents_map[v]:
                                continue
                            s = remaining_uses(x)
                            if s < best_score:
                                best_score = s
                                best = x
                        if best is not None:
                            ops.append(['delete', best])
                            pebbled.discard(best)
                        else:
                            break
            ops.append(['compute', v])
            pebbled.add(v)
        ops.append(['store', v])
        pebbled.discard(v)
    
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
