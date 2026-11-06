def pebble(dag, fast_memory_budget):
    import time
    from collections import defaultdict
    
    start_time = time.time()
    DEADLINE = 85.0
    
    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
        
        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            queue.sort()
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        topo_index = {v: i for i, v in enumerate(topo_order)}
        last_use = {}
        for v in all_vertices:
            if v in outputs:
                last_use[v] = len(topo_order)
            else:
                max_idx = -1
                for c in children.get(v, []):
                    if c in topo_index:
                        max_idx = max(max_idx, topo_index[c])
                last_use[v] = max_idx
        
        ops = []
        fast_mem = set()
        
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break
            
            ps = parents.get(v, [])
            
            if not ps:
                if v not in fast_mem:
                    ops.append(['load', v])
                    fast_mem.add(v)
            else:
                for p in ps:
                    if p not in fast_mem:
                        ops.append(['load', p])
                        fast_mem.add(p)
                
                ops.append(['compute', v])
                fast_mem.add(v)
            
            # Delete/store vertices that are no longer needed
            for p in list(fast_mem):
                if last_use.get(p, -1) == i:
                    if p in inputs:
                        ops.append(['store', p])
                    else:
                        ops.append(['delete', p])
                    fast_mem.discard(p)
            
            # Enforce fast memory budget
            while len(fast_mem) > fast_memory_budget:
                candidates = [p for p in fast_mem if p not in outputs]
                if not candidates:
                    break
                to_evict = min(candidates, key=lambda p: last_use.get(p, -1))
                if to_evict in inputs:
                    ops.append(['store', to_evict])
                else:
                    ops.append(['delete', to_evict])
                fast_mem.discard(to_evict)
        
        # Ensure outputs are in blue at the end
        for o in outputs:
            if o in fast_mem:
                ops.append(['store', o])
                fast_mem.discard(o)
        
        if len(ops) <= 20000:
            return ops
        else:
            return ops[:20000]
            
    except Exception:
        return []
