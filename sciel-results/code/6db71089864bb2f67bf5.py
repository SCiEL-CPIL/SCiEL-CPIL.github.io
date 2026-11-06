import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s deadline to be safe

    # 1. Parse DAG
    # dag structure: 
    # Assuming dag is a dict or object with 'vertices', 'edges' or similar.
    # Based on "topologically ordered parents and output vertices", 
    # let's assume dag has 'vertices' (list), 'parents' (dict), 'outputs' (list).
    
    # Since I don't have the exact structure, I'll infer from common patterns or try to access attributes.
    # If dag is a dict:
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
        inputs = set(dag.get('inputs', []))
        if not inputs:
            # Inputs are vertices with no parents
            inputs = set(v for v in vertices if not parents.get(v, []))
    else:
        # Try object attributes
        vertices = getattr(dag, 'vertices', [])
        parents = getattr(dag, 'parents', {})
        outputs = set(getattr(dag, 'outputs', []))
        inputs = set(getattr(dag, 'inputs', []))
        if not inputs:
            inputs = set(v for v in vertices if not parents.get(v, []))

    if not vertices:
        return []

    # 2. Topological Sort (Kahn's algorithm)
    in_degree = {v: 0 for v in vertices}
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            in_degree[v] += 1
            children[p].append(v)
            
    queue = [v for v in vertices if in_degree[v] == 0]
    topo_order = []
    while queue:
        # Sort queue to ensure deterministic order?
        # Not strictly necessary, but good for reproducibility
        v = queue.pop(0) 
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(vertices):
        # Cycle detected, fallback to simple list
        topo_order = vertices

    # 3. Calculate next_use for each vertex
    # next_use[v] is the index in topo_order of the first child of v, or infinity if none.
    # Actually, we need the next time v is *used* as a parent.
    # If v is a leaf, next_use is infinity.
    
    # Let's find the next index in topo_order where v appears as a parent.
    # Since topo_order is linear, we can just look at the children's positions.
    pos_in_topo = {v: i for i, v in enumerate(topo_order)}
    
    next_use = {}
    for v in vertices:
        min_pos = float('inf')
        for c in children[v]:
            if c in pos_in_topo:
                min_pos = min(min_pos, pos_in_topo[c])
        next_use[v] = min_pos

    # 4. Pebbling Simulation
    mem = set()
    stored = set()
    ops = []
    
    B = fast_memory_budget
    
    # Helper to evict
    def evict_one():
        if not mem:
            return
        # Find vertex with max next_use
        # If next_use is inf, it's not used again, so it's a good candidate for deletion.
        # But we want to keep vertices with *small* next_use in memory.
        # So we evict the one with *largest* next_use.
        
        # Tie-breaker: prefer deleting vertices with inf next_use (no future use)
        # If all have inf, delete any.
        
        candidates = list(mem)
        # Sort by next_use descending
        # If next_use is inf, treat as very large
        candidates.sort(key=lambda x: (next_use[x] == float('inf'), next_use[x]), reverse=True)
        
        w = candidates[0]
        mem.remove(w)
        
        if next_use[w] == float('inf'):
            # Not used again, delete
            ops.append(['delete', w])
        else:
            # Used again, store
            ops.append(['store', w])
            stored.add(w)
            
    # Process vertices
    for v in topo_order:
        # Check time
        if time.time() > deadline:
            break
            
        if v in inputs:
            if v not in mem:
                if len(mem) >= B:
                    evict_one()
                ops.append(['load', v])
                mem.add(v)
                # Inputs are always available in slow memory, so no need to track 'stored' for them specifically in the same way, 
                # but if we evict an input, we can just load it again.
        else:
            # Internal or Output
            # Ensure parents are in mem
            for p in parents.get(v, []):
                if p not in mem:
                    if p in stored:
                        if len(mem) >= B:
                            evict_one()
                        ops.append(['load', p])
                        mem.add(p)
                    else:
                        # p was deleted, need to recompute
                        # This is complex. To avoid, we should have stored p if next_use[p] was not inf.
                        # If we deleted p, it means next_use[p] was inf at the time of eviction.
                        # But if p is a parent of v, and v is in topo_order after p, 
                        # then p's next_use should have been <= pos_in_topo[v].
                        # So next_use[p] should NOT have been inf.
                        # Unless the topological sort order is such that p's last child is v, 
                        # and we evicted p after processing p's other children but before v?
                        # No, we process in topo_order. 
                        # If p is a parent of v, p comes before v.
                        # next_use[p] is the min pos of its children.
                        # So next_use[p] <= pos_in_topo[v].
                        # So next_use[p] is not inf.
                        # So we should have stored p.
                        # So this case (p not in stored) should not happen if logic is correct.
                        # Fallback: Load from slow memory if it was an input? 
                        # If p was an input, it's in 'inputs', not 'stored' initially.
                        # But we added it to mem. If evicted, we didn't add to stored.
                        # So if p is an input and evicted, we can just load it again.
                        if p in inputs:
                            if len(mem) >= B:
                                evict_one()
                            ops.append(['load', p])
                            mem.add(p)
                        else:
                            # Should not happen, but recompute if needed?
                            # For now, assume it's in stored or input.
                            pass

            # Compute v
            ops.append(['compute', v])
            
            # Add v to mem
            if len(mem) >= B:
                evict_one()
            mem.add(v)
            
            # If v is an output, we must ensure it ends up in slow memory.
            # We don't know when it will be evicted.
            # When it is evicted, if it's an output, we should store it.
            # But my evict_one logic stores if next_use is not inf.
            # If v is an output and has no children, next_use[v] is inf.
            # So it would be deleted.
            # But outputs must finish blue.
            # So we must force store for outputs.
            
            # Modify evict_one or handle outputs separately.
            # Let's modify the eviction logic to know about outputs.
            
    # Post-processing: Ensure all outputs are in slow memory.
    # Currently, outputs in mem need to be stored.
    # Outputs not in mem and not in stored are problematic.
    
    # Let's refine the eviction to handle outputs.
    # We'll re-run the simulation with a fixed evict function.
    
    mem = set()
    stored = set()
    ops = []
    
    def evict_one_v2():
        if not mem:
            return
        candidates = list(mem)
        # Priority:
        # 1. Vertices with next_use = inf (not used again). Among these, prefer non-outputs?
        #    Actually, if it's an output, it MUST be stored.
        #    If it's not an output, it can be deleted.
        # 2. Vertices with next_use < inf. They MUST be stored.
        
        # We want to evict the "least useful" vertex.
        # Least useful: 
        # - If next_use is inf:
        #   - If output: Must store. High cost (store).
        #   - If not output: Can delete. Low cost (delete).
        #   - Prefer deleting non-outputs with inf next_use.
        # - If next_use < inf:
        #   - Must store.
        #   - Prefer evicting the one with largest next_use (furthest in future).
        
        # Score for eviction:
        # We want to minimize impact.
        # Impact of evicting w:
        # If w is output and next_use inf: Store (1 op).
        # If w is not output and next_use inf: Delete (1 op).
        # If w is not output and next_use < inf: Store (1 op).
        # If w is output and next_use < inf: Store (1 op).
        
        # All evictions cost 1 op (store or delete).
        # But storing puts it in 'stored', allowing reload. Deleting forces recompute (expensive).
        # So we MUST store if next_use < inf.
        # If next_use is inf:
        #   If output: MUST store.
        #   If not output: Can delete.
        
        # So, candidates for deletion (next_use inf, not output) are best.
        # Candidates for store (next_use < inf OR output) are worse.
        
        # Among candidates for store, evict the one with largest next_use.
        
        del_candidates = []
        store_candidates = []
        
        for w in candidates:
            if next_use[w] == float('inf') and w not in outputs:
                del_candidates.append(w)
            else:
                store_candidates.append(w)
                
        if del_candidates:
            # Pick any, or one with largest next_use (all inf, so tie)
            w = del_candidates[0]
            mem.remove(w)
            ops.append(['delete', w])
        else:
            # Must store
            # Pick one with largest next_use
            store_candidates.sort(key=lambda x: next_use[x], reverse=True)
            w = store_candidates[0]
            mem.remove(w)
            ops.append(['store', w])
            stored.add(w)

    for v in topo_order:
        if time.time() > deadline:
            break
            
        if v in inputs:
            if v not in mem:
                if len(mem) >= B:
                    evict_one_v2()
                ops.append(['load', v])
                mem.add(v)
        else:
            # Internal or Output
            for p in parents.get(v, []):
                if p not in mem:
                    if p in stored:
                        if len(mem) >= B:
                            evict_one_v2()
                        ops.append(['load', p])
                        mem.add(p)
                    elif p in inputs:
                        if len(mem) >= B:
                            evict_one_v2()
                        ops.append(['load', p])
                        mem.add(p)
                    else:
                        # Should not happen
                        pass

            ops.append(['compute', v])
            
            if len(mem) >= B:
                evict_one_v2()
            mem.add(v)
            
    # Final step: Ensure all outputs are in slow memory.
    # Any output still in mem must be stored.
    for v in mem:
        if v in outputs:
            ops.append(['store', v])
            stored.add(v)
            mem.remove(v)
        else:
            # Not output, can delete
            ops.append(['delete', v])
            mem.remove(v)
            
    return ops
