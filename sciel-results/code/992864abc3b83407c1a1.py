"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
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
    DEADLINE = 85.0  # wall-clock limit from evaluator (90s limit, 5s margin)

    try:
        parents = dag.get('parents', {})
        outputs = set(dag.get('outputs', []))
        inputs = set(dag.get('inputs', []))
        
        if fast_memory_budget is None or fast_memory_budget < 1:
            fast_memory_budget = 1

        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | inputs | outputs
        
        # Topological sort
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
            
        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            # Use a simple sort to ensure deterministic order, or just pop
            # Sorting by vertex ID for stability
            queue.sort()
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
            
        if len(topo_order) != len(all_vertices):
             # Cycle detected or invalid DAG, fallback to simple list
             topo_order = list(all_vertices)

        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # Precompute last_use: the highest topo index of a vertex that depends on v
        # If v is an output, it must be kept until the end? 
        # The task says "all outputs must finish blue". This implies they must be in fast memory at the end.
        # So last_use[output] = len(topo_order) (conceptually, after the last step).
        
        last_use = {}
        for v in all_vertices:
            max_idx = -1
            for c in children.get(v, []):
                if c in topo_index:
                    if topo_index[c] > max_idx:
                        max_idx = topo_index[c]
            if v in outputs:
                # Outputs must be present at the end. 
                # We can treat them as needed until the very end.
                max_idx = max(max_idx, len(topo_order))
            last_use[v] = max_idx

        ops = []
        fast_mem = set()
        
        def evict_one(exclude_set):
            """Evict one vertex from fast_mem, preferring those not in exclude_set and with smallest last_use."""
            candidates = [v for v in fast_mem if v not in exclude_set]
            if not candidates:
                return None
            # Sort by last_use (ascending), then by topo_index (ascending) for stability
            candidates.sort(key=lambda v: (last_use[v], topo_index.get(v, 0)))
            victim = candidates[0]
            fast_mem.discard(victim)
            ops.append(['delete', victim])
            return victim

        for v in topo_order:
            if time.time() > DEADLINE:
                break
                
            ps = parents.get(v, [])
            
            if v in inputs:
                # Input vertex
                if v not in fast_mem:
                    # Need to load v
                    # If budget is full, evict someone
                    if len(fast_mem) >= fast_memory_budget:
                        # We cannot evict v (not in mem yet)
                        # We can evict anyone not needed for v? 
                        # Actually, if v is an input, its parents are none.
                        # So we can evict anyone.
                        evict_one(set())
                    ops.append(['load', v])
                    fast_mem.add(v)
            else:
                # Compute vertex v
                # Ensure all parents are in fast_mem
                for p in ps:
                    if p not in fast_mem:
                        if len(fast_mem) >= fast_memory_budget:
                            # Need to make space
                            # We cannot evict other parents of v that are already in mem? 
                            # Actually, we can evict any parent that is not p and not needed for v?
                            # No, we need ALL parents to compute v.
                            # So we must evict a vertex that is NOT a parent of v.
                            exclude = set(ps)
                            exclude.add(v) # v not in mem yet
                            evict_one(exclude)
                        ops.append(['load', p])
                        fast_mem.add(p)
                
                # Compute v
                ops.append(['compute', v])
                fast_mem.add(v)
                
                # Check if we exceeded budget (shouldn't if we evicted before loading, 
                # but v was added after loading parents. 
                # If budget was full, we evicted one, loaded one parent (net 0), 
                # then computed v (net +1). So we might be at budget+1 if we didn't evict enough?
                # Logic: 
                # 1. Check parents. If p not in mem:
                #    If full: Evict 1 (not parent). Load p. (Mem size stays same)
                # 2. Compute v. Add v. (Mem size increases by 1)
                # So if we were at budget, we are now at budget+1.
                # We must evict one more if len(fast_mem) > fast_memory_budget.
                
                if len(fast_mem) > fast_memory_budget:
                    # Evict a vertex that is no longer needed or least urgent
                    # Exclude v (just computed, might be needed)
                    # Exclude parents of v? No, parents are done being used for v.
                    # But they might be needed for other children.
                    # We should evict the one with smallest last_use.
                    exclude = {v}
                    evict_one(exclude)

        # Ensure all outputs are in fast memory
        # This is tricky if we evicted an output.
        # If an output is not in fast_mem, we must load it (recompute? No, load from slow).
        # But if it was computed and deleted, we can just load it.
        for o in outputs:
            if o not in fast_mem:
                if len(fast_mem) >= fast_memory_budget:
                    evict_one(set())
                ops.append(['load', o])
                fast_mem.add(o)
        
        # Final check: if we still exceed budget, evict non-outputs
        while len(fast_mem) > fast_memory_budget:
            candidates = [v for v in fast_mem if v not in outputs]
            if not candidates:
                break
            candidates.sort(key=lambda v: (last_use[v], topo_index.get(v, 0)))
            victim = candidates[0]
            fast_mem.discard(victim)
            ops.append(['delete', victim])

        if len(ops) <= 20000:
            return ops
        else:
            return ops[:20000]
            
    except Exception as e:
        # Fallback to a very simple schedule if something goes wrong
        try:
            parents = dag.get('parents', {})
            inputs = dag.get('inputs', [])
            outputs = dag.get('outputs', [])
            all_vertices = list(parents.keys()) + [v for v in inputs if v not in parents]
            ops = []
            for v in all_vertices:
                if v in inputs:
                    ops.append(['load', v])
                else:
                    for p in parents.get(v, []):
                        if p not in inputs: # Simplistic
                             ops.append(['load', p])
                    ops.append(['compute', v])
            return ops[:20000]
        except:
            return []
# EVOLVE-BLOCK-END
