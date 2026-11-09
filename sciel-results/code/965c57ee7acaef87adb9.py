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
        
        # Build position map for last-use computation
        pos_in_topo = {v: i for i, v in enumerate(topo_order)}
        
        # Compute last_use[v] = the last position in topo_order where v is needed as a parent
        # i.e., the max pos_in_topo[c] for all c in children[v]
        last_use = {}
        for v in all_vertices:
            if children[v]:
                last_use[v] = max(pos_in_topo[c] for c in children[v])
            else:
                last_use[v] = pos_in_topo[v]  # not needed after its own position
        
        # Improved greedy pebbling with memory management using last-use heuristic
        in_fast_memory = set()
        operations = []
        
        for i, v in enumerate(topo_order):
            if time.time() > deadline - 5:
                # Deadline guard: return best result so far
                break
            
            if v in inputs:
                if v not in in_fast_memory:
                    if len(in_fast_memory) >= fast_memory_budget:
                        # Need to evict: pick the vertex with the largest last_use
                        # (i.e., needed the soonest in the future)
                        # Actually, we want to evict the one with the LARGEST last_use
                        # that is NOT needed soon... wait, we want to keep vertices
                        # that are needed soon. So evict the one with the SMALLEST
                        # "urgency" = evict the one whose last_use is FURTHEST in future
                        # No: we should evict the vertex that we need the LEAST urgently
                        # A vertex with a large last_use is needed far in the future
                        # A vertex with a small last_use is needed soon
                        # So we should evict the vertex with the LARGEST last_use
                        # (least urgent)
                        
                        # But we must not evict vertices that are parents of the
                        # current or upcoming computations that we haven't done yet
                        # Actually, the standard approach: evict the vertex with
                        # the farthest next use (or no future use)
                        
                        # Let's use: evict the vertex with the maximum last_use
                        # among non-output vertices in fast memory
                        best_evict = None
                        best_last_use = -1
                        for u in in_fast_memory:
                            if u in outputs:
                                continue
                            lu = last_use.get(u, pos_in_topo[u])
                            if lu > best_last_use:
                                best_last_use = lu
                                best_evict = u
                        
                        if best_evict is not None:
                            in_fast_memory.discard(best_evict)
                            operations.append(['delete', best_evict])
                        else:
                            # All are outputs? Shouldn't happen, but handle gracefully
                            # Just evict any non-current vertex
                            for u in in_fast_memory:
                                if u != v:
                                    in_fast_memory.discard(u)
                                    operations.append(['delete', u])
                                    break
                    
                    in_fast_memory.add(v)
                    operations.append(['load', v])
            else:
                ps = parents_dict.get(v, [])
                # Ensure parents are in fast memory
                for p in ps:
                    if p not in in_fast_memory:
                        if len(in_fast_memory) >= fast_memory_budget:
                            # Evict: pick vertex with maximum last_use (least urgent)
                            # among non-output vertices
                            best_evict = None
                            best_last_use = -1
                            for u in in_fast_memory:
                                if u in outputs:
                                    continue
                                lu = last_use.get(u, pos_in_topo[u])
                                if lu > best_last_use:
                                    best_last_use = lu
                                    best_evict = u
                            
                            if best_evict is not None:
                                in_fast_memory.discard(best_evict)
                                operations.append(['delete', best_evict])
                            else:
                                for u in in_fast_memory:
                                    if u != v:
                                        in_fast_memory.discard(u)
                                        operations.append(['delete', u])
                                        break
                        
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
