def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    # Extract DAG info
    try:
        if isinstance(dag, dict):
            parents_dict = dict(dag.get('parents', {}))
            inputs = list(dag.get('inputs', []))
            outputs = set(dag.get('outputs', []))
        else:
            parents_dict = dict(getattr(dag, 'parents', {}))
            inputs = list(getattr(dag, 'inputs', []))
            outputs = set(getattr(dag, 'outputs', []))
    except Exception:
        return []

    # Build graph structure
    all_vertices = set(inputs)
    for v, ps in parents_dict.items():
        all_vertices.add(v)
        for p in ps:
            all_vertices.add(p)

    children = {v: [] for v in all_vertices}
    for v, ps in parents_dict.items():
        for p in ps:
            if p in children:
                children[p].append(v)

    # Compute topological order
    from collections import deque
    in_degree = {v: len(parents_dict.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    # If topo order doesn't cover all vertices, something is wrong
    if len(topo_order) != len(all_vertices):
        return []

    # Compute future uses: for each vertex, how many times it will be needed
    # after its current position in topo order
    future_uses = {v: 0 for v in all_vertices}
    # Count total uses (number of children)
    total_uses = {v: len(children[v]) for v in all_vertices}
    
    # For each vertex in reverse topo order, track remaining uses
    # Actually, let's compute "remaining uses" more carefully
    # A vertex v is used by its children. When we process v in topo order,
    # all its children come later. So remaining uses = total_uses[v]
    # But if we recompute, we might use it again.
    # For the greedy, we just need to know: after computing v, will any of its
    # children still need v? Yes, all children of v need v.
    # So remaining_uses[v] = len(children[v]) for the purpose of deciding
    # whether to keep v in memory.

    # Better approach: compute for each vertex, the set of descendants that need it
    # Actually for the greedy, we just need: after processing v, does any future
    # vertex need v? That's simply len(children[v]) > 0.
    # But wait - if a child c of v is processed, it needs v. After c is processed,
    # does c's child need v? No, c's child needs c, not v (unless v is also a parent).
    # So v is needed exactly by its direct children.

    # Let's use a different approach: count remaining references
    remaining_refs = {v: 0 for v in all_vertices}
    for v in all_vertices:
        for c in children[v]:
            remaining_refs[v] += 1

    # Now the greedy pebbling with better eviction strategy
    in_fast_memory = set()
    operations = []

    def time_ok():
        return (time.time() - start_time) < deadline

    for v in topo_order:
        if not time_ok():
            break

        if v in inputs:
            if v not in in_fast_memory:
                # Need to load input v
                if len(in_fast_memory) >= fast_memory_budget:
                    # Need to evict something
                    # Find best candidate to evict
                    best_evict = None
                    best_score = float('inf')
                    for u in in_fast_memory:
                        if u in outputs:
                            continue  # Can't evict outputs
                        # Score: prefer evicting vertices with fewer remaining refs
                        # and vertices that are inputs (can be reloaded)
                        score = remaining_refs.get(u, 0)
                        if u in inputs:
                            score -= 0.5  # Slight preference for evicting inputs
                        if score < best_score:
                            best_score = score
                            best_evict = u
                    
                    if best_evict is not None:
                        in_fast_memory.discard(best_evict)
                        operations.append(['delete', best_evict])
                
                in_fast_memory.add(v)
                operations.append(['load', v])
        else:
            ps = parents_dict.get(v, [])
            # Ensure all parents are in fast memory
            for p in ps:
                if p not in in_fast_memory:
                    # Need to load parent p
                    if len(in_fast_memory) >= fast_memory_budget:
                        # Evict best candidate
                        best_evict = None
                        best_score = float('inf')
                        for u in in_fast_memory:
                            if u in outputs:
                                continue
                            score = remaining_refs.get(u, 0)
                            if u in inputs:
                                score -= 0.5
                            if score < best_score:
                                best_score = score
                                best_evict = u
                        
                        if best_evict is not None:
                            in_fast_memory.discard(best_evict)
                            operations.append(['delete', best_evict])
                    
                    in_fast_memory.add(p)
                    operations.append(['load', p])
            
            # Compute v
            in_fast_memory.add(v)
            operations.append(['compute', v])
        
        # After processing v, decrement remaining refs for its parents
        # Actually, we should decrement when the consumer is done
        # For each parent of v, one reference is consumed
        if v not in inputs:
            for p in parents_dict.get(v, []):
                remaining_refs[p] = max(0, remaining_refs.get(p, 0) - 1)
        
        # Clean up: if a vertex is no longer needed and not an output, delete it
        # But be careful: we should only delete if it's safe
        for u in list(in_fast_memory):
            if u in outputs:
                continue
            if remaining_refs.get(u, 0) == 0:
                # u is no longer needed
                in_fast_memory.discard(u)
                operations.append(['delete', u])

    # Store all outputs
    for v in outputs:
        if not time_ok():
            break
        if v in in_fast_memory:
            in_fast_memory.discard(v)
            operations.append(['store', v])
        else:
            # This shouldn't happen if we handled it correctly,
            # but just in case, we need to ensure output is available
            # For now, if it's not in memory, we have a problem
            # The output should have been computed and kept until stored
            pass

    # Check if we have too many operations
    if len(operations) > 20000:
        # Truncate - this is a fallback
        # Try to return a valid subset
        # For safety, just return the baseline-style approach
        pass

    return operations
