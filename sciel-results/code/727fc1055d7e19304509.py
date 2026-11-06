"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    dag structure expected:
      - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
      - dag['outputs']: list of output vertices
      - dag['inputs']: list of input vertices (start blue)
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}.
    """
    start_time = time.time()
    DEADLINE = 85.0  # wall-clock limit from evaluator with safety margin
    
    # Try to import baseline as fallback
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        
        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        # Topological order
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
        
        # Map vertex to index in topo_order for O(1) lookup
        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # Precompute: for each vertex, the set of children (direct dependents)
        # and the "last_use" index: the highest topo_index of any child
        last_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                last_use[v] = max(topo_index[c] for c in children[v])
            else:
                last_use[v] = topo_index.get(v, -1)
        
        # Also ensure outputs are kept until the end
        must_be_blue = set(outputs)
        
        # Strategy: Process in topological order, but use a smarter eviction policy.
        # We keep vertices in fast memory. When memory is full (or to minimize transfers),
        # we evict the vertex with the latest last_use (LIFO-like for memory pressure) 
        # OR we can use a heuristic based on "urgency".
        
        # Actually, a better heuristic for pebbling is to minimize the number of loads.
        # A vertex needs to be loaded if it's an input (once) or if it was deleted and is needed again.
        # To minimize loads, we should keep a vertex as long as possible if it has many future uses,
        # or if recomputing it would be expensive (high depth/fan-in).
        
        # Let's use a priority queue for eviction.
        # When we need to free space (if fast_memory_budget is small) or just to be clean,
        # we delete vertices that are no longer needed (last_use < current_index).
        
        # However, the problem statement says "every red value consumes one fast-memory slot".
        # It doesn't explicitly say we MUST stay within fast_memory_budget at all times?
        # "under a bounded candidate interface... fast_memory_budget".
        # Usually, this implies we cannot exceed the budget.
        # If we exceed, it's invalid.
        
        # Let's assume we must respect the budget.
        # If fast_memory_budget is large enough to hold all live vertices, we just keep them.
        # If not, we must evict.
        
        # Heuristic for eviction:
        # When adding a new vertex (compute or load) causes |fast_mem| > fast_memory_budget,
        # we must delete some vertex.
        # Which one?
        # 1. It must be safe to delete (not an output that hasn't been stored yet? No, outputs must finish blue, so they must be in fast mem at the end. If we delete an output, we must reload it later. That's a transfer. So avoid deleting outputs unless necessary.)
        # 2. Prefer deleting vertices with the latest last_use? No, that's LIFO.
        # 3. Prefer deleting vertices with the earliest last_use? No, that's FIFO.
        # 4. Prefer deleting vertices that are "easiest" to recompute?
        #    Recomputation cost is roughly the number of ancestors.
        #    So delete the vertex with the smallest "recomputation cost" (shallowest) among those that are not outputs.
        
        # Let's precompute depth (longest path from input) or just use topological depth.
        # depth[v] = max(depth[p] for p in parents[v]) + 1
        depth = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1
        
        # We will maintain a list of (last_use, depth, vertex) in fast_mem to decide eviction.
        # Actually, we can just iterate through fast_mem and pick the best candidate.
        # Given the constraint of 20000 ops and likely small DAGs, this is fine.
        
        ops = []
        fast_mem = set()
        
        # To optimize, we can pre-plan the sequence.
        # But let's stick to the online greedy approach with a good eviction policy.
        
        def get_eviction_candidate():
            """
            Find the vertex in fast_mem to delete to make space.
            Criteria:
            1. Not an output (unless all are outputs, then pick one with latest last_use? No, outputs must be blue at end. If we delete, we must reload. So avoid if possible.)
            2. Among non-outputs, pick the one with the smallest depth (easiest to recompute).
            3. Tie-breaker: earliest last_use (less useful in the near future).
            """
            candidates = []
            for v in fast_mem:
                if v in must_be_blue:
                    continue
                # Only consider vertices that are actually needed in the future?
                # If last_use[v] < current_index, it's already "dead" and should have been deleted.
                # So we should delete "dead" vertices first before evicting live ones.
                candidates.append((depth[v], last_use[v], v))
            
            if not candidates:
                # All in fast_mem are outputs. We have to delete one.
                # Pick the one with the latest last_use? Or earliest?
                # If we delete an output, we must reload it at the end.
                # It's better to delete the one that is "most expensive" to keep?
                # Or the one that is least likely to be needed soon?
                # Since they are outputs, they are needed at the very end.
                # So all have last_use >= current_index (conceptually, until the end).
                # Let's pick the one with the largest depth (hardest to recompute) to KEEP?
                # No, we want to DELETE.
                # We want to delete the one that is EASIEST to recompute (smallest depth).
                candidates = [(depth[v], last_use[v], v) for v in fast_mem]
            
            if not candidates:
                return None
            
            # Sort by depth (ascending), then last_use (ascending)
            candidates.sort(key=lambda x: (x[0], x[1]))
            return candidates[0][2]

        # Main loop
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break
            
            ps = parents.get(v, [])
            
            if not ps:
                # Input vertex
                if v not in fast_mem:
                    # Check budget
                    if len(fast_mem) >= fast_memory_budget:
                        # Need to evict
                        # But inputs are usually cheap to reload?
                        # Actually, inputs are just data. Reloading is a 'load' op.
                        # Recomputing an input is impossible (it's a source).
                        # So we MUST load it.
                        # If we are full, we must evict something else.
                        cand = get_eviction_candidate()
                        if cand:
                            ops.append(['delete', cand])
                            fast_mem.discard(cand)
                    
                    ops.append(['load', v])
                    fast_mem.add(v)
            else:
                # Compute vertex
                # Ensure all parents are in fast_mem
                for p in ps:
                    if p not in fast_mem:
                        # Check budget
                        if len(fast_mem) >= fast_memory_budget:
                            cand = get_eviction_candidate()
                            if cand:
                                ops.append(['delete', cand])
                                fast_mem.discard(cand)
                        
                        # Load parent
                        # Note: If p is an input, it's a 'load'. If p is computed, it's also a 'load' (from slow memory) or 'compute' (if we recompute).
                        # The problem says "Recomputations allowed".
                        # 'load' implies taking from slow memory.
                        # 'compute' implies calculating from parents.
                        # If p was deleted, its value is gone from fast memory.
                        # Can we 'load' it? Only if it was stored?
                        # "Inputs start blue... every red value consumes one fast-memory slot".
                        # Usually in pebbling, 'load' means bringing from slow memory to fast.
                        # If we deleted a computed vertex, did we store it?
                        # The operations are load/store/compute/delete.
                        # If we just 'delete', the value is lost.
                        # So we must 'store' before 'delete' if we want to 'load' later?
                        # Or does 'delete' just remove the pebble, and the value is gone?
                        # "Recomputations allowed" suggests we can compute it again from its parents.
                        # So if p is not in fast_mem, we can either:
                        # 1. 'load' p (if it was stored previously? The problem doesn't explicitly mention a store step in the flow, but 'store' is an operation).
                        # 2. 'compute' p (if its parents are available).
                        
                        # Let's assume 'load' is only valid for inputs or previously stored values.
                        # To be safe, if p is not an input and not in fast_mem, we should RECOMPUTE it.
                        # Recomputing p requires its parents to be in fast_mem.
                        # This could lead to a cascade.
                        
                        # Simplification:
                        # If p is an input: 'load' p.
                        # If p is not an input: We need to recompute p.
                        # But recomputing p is just "computing" it again.
                        # So we can just call a helper to "ensure p is in fast_mem".
                        
                        pass # Handled below in a recursive/iterative ensure function
        
        # The simple loop above is flawed because it doesn't handle recomputation of non-input parents correctly.
        # Let's rewrite the core logic with a helper function.
        
        ops = []
        fast_mem = set()
        
        def ensure_in_fast_mem(v):
            """
            Ensure vertex v is in fast_mem.
            Returns True if successful, False if budget exceeded and couldn't free space.
            """
            if v in fast_mem:
                return True
            
            ps = parents.get(v, [])
            if not ps:
                # It's an input
                if len(fast_mem) >= fast_memory_budget:
                    cand = get_eviction_candidate()
                    if cand is None:
                        return False
                    ops.append(['delete', cand])
                    fast_mem.discard(cand)
                
                ops.append(['load', v])
                fast_mem.add(v)
                return True
            else:
                # It's a computed vertex
                # First, ensure all its parents are in fast_mem
                for p in ps:
                    if not ensure_in_fast_mem(p):
                        return False
                
                # Now compute v
                if len(fast_mem) >= fast_memory_budget:
                    cand = get_eviction_candidate()
                    if cand is None:
                        return False
                    ops.append(['delete', cand])
                    fast_mem.discard(cand)
                
                ops.append(['compute', v])
                fast_mem.add(v)
                return True
        
        # Process in topological order
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break
            
            if not ensure_in_fast_mem(v):
                # Failed to get v in fast mem, break?
                # This shouldn't happen if budget is reasonable or we evict correctly.
                break
            
            # Optional: Delete vertices that are no longer needed to save memory for future?
            # We can proactively delete vertices whose last_use < i (current index).
            # This keeps fast_mem small and reduces pressure.
            for u in list(fast_mem):
                if u == v:
                    continue
                if last_use[u] < i:
                    # u is no longer needed
                    ops.append(['delete', u])
                    fast_mem.discard(u)
        
        # Ensure all outputs are in fast_mem at the end
        for o in outputs:
            if o not in fast_mem:
                if not ensure_in_fast_mem(o):
                    # Try to evict and load
                    if len(fast_mem) >= fast_memory_budget:
                        cand = get_eviction_candidate()
                        if cand:
                            ops.append(['delete', cand])
                            fast_mem.discard(cand)
                    
                    # If it's an input, load. If computed, compute.
                    # ensure_in_fast_mem handles this, but if it failed due to budget,
                    # we might need to force evict an output?
                    # If ensure_in_fast_mem failed, it means we couldn't free space.
                    # This implies fast_memory_budget is too small to hold all necessary live vars + outputs.
                    # In that case, we might have to delete an output to make room, then reload it?
                    # That's a net zero gain in memory, but adds ops.
                    
                    # Let's just try ensure again, maybe it works now?
                    ensure_in_fast_mem(o)
        
        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            # Too many ops, try baseline or truncate
            if pebble_baseline:
                return pebble_baseline(dag, fast_memory_budget)
            return ops[:20000]
            
    except Exception as e:
        # Fallback to baseline if available
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
# EVOLVE-BLOCK-END
