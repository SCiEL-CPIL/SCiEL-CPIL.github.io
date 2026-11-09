def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Fallback to baseline
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        baseline_ops = pebble_baseline(dag, fast_memory_budget)
    except:
        baseline_ops = None
    
    # Try to improve
    try:
        # Extract DAG info
        if isinstance(dag, dict):
            parents_dict = dict(dag.get('parents', {}))
            inputs = list(dag.get('inputs', []))
            outputs = set(dag.get('outputs', []))
        else:
            parents_dict = dict(getattr(dag, 'parents', {}))
            inputs = list(getattr(dag, 'inputs', []))
            outputs = set(getattr(dag, 'outputs', []))
        
        # Compute topological order
        all_vertices = set(inputs)
        for v, ps in parents_dict.items():
            all_vertices.add(v)
            all_vertices.update(ps)
        
        in_degree = {v: len(parents_dict.get(v, [])) for v in all_vertices}
        children = {v: [] for v in all_vertices}
        for v, ps in parents_dict.items():
            for p in ps:
                children[p].append(v)
        
        from collections import deque
        queue = deque([v for v in all_vertices if in_degree[v] == 0])
        topo_order = []
        while queue:
            v = queue.popleft()
            topo_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        # Build position map
        pos_in_topo = {v: i for i, v in enumerate(topo_order)}
        
        # Compute last_use[v] = the last position in topo_order where v is needed as a parent
        last_use = {}
        for v in all_vertices:
            if children[v]:
                last_use[v] = max(pos_in_topo[c] for c in children[v])
            else:
                last_use[v] = pos_in_topo[v]
        
        # Compute next_use[v] for vertices at position i: the earliest future position where v is needed
        # For a vertex u, next_use_from[i] = min(pos_in_topo[c] for c in children[u] if pos_in_topo[c] > i)
        # We'll compute this on the fly or precompute a sorted list of uses per vertex
        
        # Precompute for each vertex, the sorted list of positions where it's needed as a parent
        uses_of = {v: sorted([pos_in_topo[c] for c in children[v]]) for v in all_vertices}
        
        def get_next_use(u, current_pos):
            """Get the next position after current_pos where u is needed."""
            uses = uses_of.get(u, [])
            for pos in uses:
                if pos > current_pos:
                    return pos
            return float('inf')
        
        # Precompute reload cost: number of loads needed to recompute a vertex from scratch
        # For input vertices: cost = 1 (just load)
        # For computed vertices: cost = 1 + sum of costs of parents that need to be loaded
        # We'll compute this recursively with memoization
        
        reload_cost = {}
        def compute_reload_cost(v, visiting=None):
            if visiting is None:
                visiting = set()
            if v in reload_cost:
                return reload_cost[v]
            if v in visiting:
                return 1  # Cycle guard (shouldn't happen in DAG)
            visiting.add(v)
            if v in inputs:
                reload_cost[v] = 1
            else:
                ps = parents_dict.get(v, [])
                cost = 1 + sum(compute_reload_cost(p, visiting) for p in ps)
                reload_cost[v] = cost
            visiting.discard(v)
            return reload_cost[v]
        
        for v in all_vertices:
            compute_reload_cost(v)
        
        # Greedy pebbling with cost-aware eviction
        in_fast_memory = set()
        operations = []
        
        def evict_one(current_pos, exclude=None):
            """Evict the vertex with the worst (cost / next_use_urgency) ratio."""
            best_evict = None
            best_score = float('-inf')
            for u in in_fast_memory:
                if u in outputs:
                    continue
                if exclude and u == exclude:
                    continue
                nu = get_next_use(u, current_pos)
                rc = reload_cost.get(u, 1)
                # Higher cost and farther next use = better candidate for eviction
                # Use a smooth surrogate: score = rc * (nu + 1) to avoid division by zero
                # Vertices with no future use (nu=inf) get very high score
                if nu == float('inf'):
                    score = rc * 1e9
                else:
                    score = rc * (nu + 1)
                if score > best_score:
                    best_score = score
                    best_evict = u
            
            if best_evict is not None:
                in_fast_memory.discard(best_evict)
                operations.append(['delete', best_evict])
            else:
                # Fallback: evict any non-excluded vertex
                for u in in_fast_memory:
                    if exclude and u == exclude:
                        continue
                    in_fast_memory.discard(u)
                    operations.append(['delete', u])
                    break
        
        for i, v in enumerate(topo_order):
            if time.time() > deadline - 3:
                break
            
            if v in inputs:
                if v not in in_fast_memory:
                    if len(in_fast_memory) >= fast_memory_budget:
                        evict_one(i, exclude=v)
                    in_fast_memory.add(v)
                    operations.append(['load', v])
            else:
                ps = parents_dict.get(v, [])
                # Ensure parents are in fast memory
                for p in ps:
                    if p not in in_fast_memory:
                        if len(in_fast_memory) >= fast_memory_budget:
                            evict_one(i, exclude=p)
                        in_fast_memory.add(p)
                        operations.append(['load', p])
                
                in_fast_memory.add(v)
                operations.append(['compute', v])
            
            # Delete if no longer needed and not an output
            if last_use.get(v, pos_in_topo[v]) <= i and v not in outputs:
                if v in in_fast_memory:
                    in_fast_memory.discard(v)
                    operations.append(['delete', v])
            elif v in outputs:
                if v in in_fast_memory:
                    in_fast_memory.discard(v)
                    operations.append(['store', v])
        
        # Ensure all outputs are stored
        for v in outputs:
            if v in in_fast_memory:
                in_fast_memory.discard(v)
                operations.append(['store', v])
        
        # Count transfers
        my_transfers = sum(1 for op in operations if op[0] in ['load', 'store'])
        
        # Compare with baseline
        if baseline_ops is not None:
            baseline_transfers = sum(1 for op in baseline_ops if op[0] in ['load', 'store'])
            if my_transfers < baseline_transfers:
                return operations
            else:
                return baseline_ops
        else:
            return operations
    
    except:
        if baseline_ops is not None:
            return baseline_ops
        return []
