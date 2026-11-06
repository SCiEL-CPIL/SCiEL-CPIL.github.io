<<<<<<< SEARCH
def _pebble_greedy_memory_aware(dag, budget):
    """Greedy that respects memory budget, loading/computing as needed."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices, vset)
    
    ops = []
    in_mem = set()
    stored = set()
    
    def ensure_in_mem(v):
        if v in in_mem:
            return
        ps = parents.get(v, [])
        if not ps:
            if len(in_mem) >= b:
                for x in list(in_mem):
                    if x not in outset:
                        ops.append(["delete", x])
                        in_mem.discard(x)
                        if len(in_mem) < b:
                            break
            ops.append(["load", v])
            in_mem.add(v)
        else:
            for p in ps:
                ensure_in_mem(p)
            ops.append(["compute", v])
            in_mem.add(v)
    
    for v in outputs:
        ensure_in_mem(v)
        ops.append(["store", v])
        stored.add(v)
        in_mem.discard(v)
        to_delete = []
        for x in in_mem:
            if x not in outset:
                needed = False
                for rem_out in outputs:
                    if rem_out in stored:
                        continue
                    # Check if x is an ancestor of rem_out
                    stack = [x]
                    while stack:
                        u = stack.pop()
                        if u == rem_out:
                            needed = True
                            break
                        for ch in children.get(u, []):
                            stack.append(ch)
                    if needed:
                        break
                if not needed:
                    to_delete.append(x)
        for x in to_delete:
            ops.append(["delete", x])
            in_mem.discard(x)
    
    return ops[:20000]
=======
def _pebble_greedy_memory_aware(dag, budget):
    """Greedy that respects memory budget, loading/computing as needed."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    vset = set(vertices)
    outset = set(outputs)
    b = max(1, int(budget) if budget is not None else len(vertices))
    
    topo, children = _topo_order(parents, vertices, vset)
    
    # Precompute which vertices are ancestors of which outputs
    # ancestor_outputs[v] = set of outputs that have v as an ancestor
    ancestor_outputs = defaultdict(set)
    for out in outputs:
        # BFS/DFS to find all ancestors
        stack = [out]
        visited = set()
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            ancestor_outputs[u].add(out)
            for p in parents.get(u, []):
                stack.append(p)
    
    ops = []
    in_mem = set()
    stored = set()
    
    def make_room():
        """Evict vertices that are no longer needed to make room."""
        while len(in_mem) >= b:
            # Find a vertex that can be safely evicted
            evictable = []
            for v in in_mem:
                if v in outset or v in stored:
                    continue
                # Check if v is needed by any remaining output
                needed = False
                for out in ancestor_outputs.get(v, set()):
                    if out not in stored:
                        needed = True
                        break
                if not needed:
                    evictable.append(v)
            
            if not evictable:
                break
            
            # Evict the first one
            v_to_del = evictable[0]
            ops.append(["delete", v_to_del])
            in_mem.discard(v_to_del)
    
    def ensure_in_mem(v):
        if v in in_mem:
            return
        ps = parents.get(v, [])
        if not ps:
            if len(in_mem) >= b:
                make_room()
            ops.append(["load", v])
            in_mem.add(v)
        else:
            for p in ps:
                if p not in in_mem:
                    ensure_in_mem(p)
            
            if len(in_mem) >= b and v not in in_mem:
                make_room()
            
            ops.append(["compute", v])
            in_mem.add(v)
    
    for v in outputs:
        ensure_in_mem(v)
        ops.append(["store", v])
        stored.add(v)
        in_mem.discard(v)
        
        # Clean up: delete vertices no longer needed
        to_delete = []
        for x in in_mem:
            if x not in outset:
                needed = False
                for out in ancestor_outputs.get(x, set()):
                    if out not in stored:
                        needed = True
                        break
                if not needed:
                    to_delete.append(x)
        for x in to_delete:
            ops.append(["delete", x])
            in_mem.discard(x)
    
    return ops[:20000]
>>>>>>> REPLACE

