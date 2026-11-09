def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 85
    
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
        topo_pos = {v: i for i, v in enumerate(topo_order)}
        
        # Pre-compute usage indices for dynamic tracking
        # uses[v] = list of indices in topo_order where v is a parent of some vertex
        # We need to know when v is *needed*.
        # v is needed when we are about to compute a child of v.
        # So, for each vertex c in topo_order, its parents are needed at index topo_pos[c].
        
        usage_indices = {v: [] for v in all_vertices}
        for i, c in enumerate(topo_order):
            if c in inputs:
                continue
            ps = parents_dict.get(c, [])
            for p in ps:
                usage_indices[p].append(i)
        
        # For inputs, they are "needed" when they are loaded? No, they are loaded once.
        # But if they are used later, they are needed then.
        # Actually, the "need" is for the *value*.
        # If v is an input, it's loaded at the start or when first needed.
        # Let's assume standard pebbling: inputs are loaded, computed vertices are computed.
        
        # We track the "next use" pointer for each vertex.
        # next_use_idx[v] = index in usage_indices[v] of the next time v is needed.
        next_use_idx = {v: 0 for v in all_vertices}
        
        in_fast_memory = set()
        operations = []
        input_set = set(inputs)
        
        def get_next_use_pos(v):
            """Returns the position in topo_order of the next time v is needed, or inf."""
            idx = next_use_idx[v]
            if idx < len(usage_indices[v]):
                return usage_indices[v][idx]
            return float('inf')
        
        def get_remaining_uses(v):
            idx = next_use_idx[v]
            return len(usage_indices[v]) - idx
        
        def make_space_if_needed():
            """Make space in fast memory if at budget. Returns True if space was made."""
            if len(in_fast_memory) < fast_memory_budget:
                return True
            
            # Find best vertex to evict
            candidates = [u for u in in_fast_memory if u not in outputs]
            if not candidates:
                return False
            
            # Sort by: 
            # 1. Largest next_use_pos (least urgent)
            # 2. Smallest remaining_uses (least valuable)
            # 3. Vertex name (for determinism)
            
            def eviction_key(u):
                nup = get_next_use_pos(u)
                ru = get_remaining_uses(u)
                # We want to evict the one with MAX nup.
                # If nup is inf, it's a very good candidate.
                # If nup is small, it's a bad candidate.
                # So we sort by -nup? No, inf is large.
                # Let's map inf to a very large number.
                # Actually, we just want the max of (nup, -ru).
                # Wait, if nup is inf, it's the max.
                # If nup is 10, and another is 5, we prefer to evict the one with 10.
                # So we want to find the candidate with the MAX (nup, -ru).
                # But we want to evict the *worst* candidate to keep.
                # The "worst" to keep is the one with the furthest next use.
                # So we select the argmax of (nup, -ru).
                return (nup, -ru)
            
            # Find the candidate with the highest eviction_key
            best_candidate = None
            best_key = None
            for u in candidates:
                key = eviction_key(u)
                if best_key is None or key > best_key:
                    best_key = key
                    best_candidate = u
            
            if best_candidate is not None:
                in_fast_memory.discard(best_candidate)
                operations.append(['delete', best_candidate])
                return True
            return False

        for i, v in enumerate(topo_order):
            if v in input_set:
                if v not in in_fast_memory:
                    if make_space_if_needed():
                        in_fast_memory.add(v)
                        operations.append(['load', v])
            else:
                ps = parents_dict.get(v, [])
                # Ensure parents are in fast memory
                for p in ps:
                    if p not in in_fast_memory:
                        if make_space_if_needed():
                            in_fast_memory.add(p)
                            operations.append(['load', p])
                
                in_fast_memory.add(v)
                operations.append(['compute', v])
            
            # Update next_use_idx for v?
            # v is "used" at this step i if it is a parent of v?
            # No, v is the vertex being processed.
            # The parents of v are used at step i.
            # So we should advance the next_use_idx for the parents of v.
            if v not in input_set:
                for p in parents_dict.get(v, []):
                    # p is used at step i
                    # Advance p's next_use_idx
                    # Ensure next_use_idx[p] points to i
                    # It should, if we initialized correctly.
                    # Just increment it.
                    next_use_idx[p] += 1
            
            # Clean up: delete vertices that are no longer needed
            # If a vertex has no more future uses, we can delete it.
            # But we only delete if we need space? 
            # Proactive deletion saves memory slots for other vertices.
            # If v is not an output and has no more uses, delete it.
            if v not in outputs:
                if get_remaining_uses(v) == 0 and v in in_fast_memory:
                    in_fast_memory.discard(v)
                    operations.append(['delete', v])
            else:
                # Store outputs
                if v in in_fast_memory:
                    in_fast_memory.discard(v)
                    operations.append(['store', v])
        
        # Final cleanup: store any remaining outputs
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
