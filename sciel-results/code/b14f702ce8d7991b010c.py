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

        # Build position map for temporal distance calculations
        pos_in_topo = {v: i for i, v in enumerate(topo_order)}

        # Pre-compute per-vertex metadata
        future_uses = {v: 0 for v in all_vertices}
        last_use_pos = {v: -1 for v in all_vertices}
        reload_cost = {v: 0 for v in all_vertices}
        next_use_pos_map = {v: {} for v in all_vertices}  # vertex -> {current_pos: next_use_pos}

        for v in topo_order:
            cs = children.get(v, [])
            future_uses[v] = len(cs)
            if cs:
                last_use_pos[v] = max(pos_in_topo[c] for c in cs)
                # Pre-compute next use positions for each possible current position
                child_positions = sorted([pos_in_topo[c] for c in cs])
                for i, cp in enumerate(child_positions):
                    for pos in range(cp):
                        if pos not in next_use_pos_map[v] or cp < next_use_pos_map[v][pos]:
                            next_use_pos_map[v][pos] = cp

            ps = parents_dict.get(v, [])
            if v in inputs:
                reload_cost[v] = 1  # Just one load operation
            else:
                reload_cost[v] = 1 + sum(1 for p in ps if p not in inputs)

        # Greedy pebbling with cost-aware memory management
        in_fast_memory = set()
        operations = []
        op_count = 0

        def evict_best(in_fast_memory_set, current_pos):
            """Find the best vertex to evict based on cost-aware scoring."""
            best_evict = None
            best_score = float('inf')

            for u in in_fast_memory_set:
                if u in outputs:
                    continue  # Never evict outputs

                fu = future_uses.get(u, 0)
                if fu == 0:
                    # No future use - great candidate
                    score = 0
                else:
                    # Compute temporal distance to next use
                    # We need to find the next child after current_pos
                    cs = children.get(u, [])
                    next_use_pos = None
                    for c in cs:
                        cp = pos_in_topo.get(c, 0)
                        if cp > current_pos:
                            if next_use_pos is None or cp < next_use_pos:
                                next_use_pos = cp

                    if next_use_pos is None:
                        # All uses are in the past
                        score = 0
                    else:
                        temporal_dist = next_use_pos - current_pos
                        # Higher temporal distance = safer to evict
                        # Higher reload cost = more expensive to evict
                        # Score: we want to evict vertices that are far away AND cheap to reload
                        # Use inverse: lower score = better to evict
                        score = -temporal_dist / (reload_cost.get(u, 1) + 1)

                if score < best_score:
                    best_score = score
                    best_evict = u

            return best_evict

        for v in topo_order:
            if time.time() > deadline:
                break

            current_pos = pos_in_topo.get(v, 0)

            if v in inputs:
                if v not in in_fast_memory:
                    # Need to load input
                    if len(in_fast_memory) >= fast_memory_budget:
                        # Find best vertex to evict
                        best_evict = evict_best(in_fast_memory, current_pos)
                        if best_evict is not None:
                            in_fast_memory.discard(best_evict)
                            operations.append(['delete', best_evict])
                            op_count += 1
                    in_fast_memory.add(v)
                    operations.append(['load', v])
                    op_count += 1
            else:
                ps = parents_dict.get(v, [])
                # Ensure parents are in fast memory
                for p in ps:
                    if p not in in_fast_memory:
                        # Need to free space if at budget
                        if len(in_fast_memory) >= fast_memory_budget:
                            # Find best vertex to evict
                            best_evict = evict_best(in_fast_memory, current_pos)
                            if best_evict is not None:
                                in_fast_memory.discard(best_evict)
                                operations.append(['delete', best_evict])
                                op_count += 1

                        in_fast_memory.add(p)
                        operations.append(['load', p])
                        op_count += 1

                in_fast_memory.add(v)
                operations.append(['compute', v])
                op_count += 1

            # Delete if no longer needed and not an output
            if future_uses.get(v, 0) == 0 and v not in outputs:
                if v in in_fast_memory:
                    in_fast_memory.discard(v)
                    operations.append(['delete', v])
                    op_count += 1
            elif v in outputs:
                if v in in_fast_memory:
                    in_fast_memory.discard(v)
                    operations.append(['store', v])
                    op_count += 1

            if op_count > 19000:
                break

        # Ensure all outputs are stored
        for v in outputs:
            if v in in_fast_memory:
                in_fast_memory.discard(v)
                operations.append(['store', v])
                op_count += 1

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
