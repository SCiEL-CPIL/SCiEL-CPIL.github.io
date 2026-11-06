# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling with refined greedy heuristics.
    
    Strategy:
    1. Parse the DAG structure
    2. Greedy pebbling with smart eviction based on future usage
    3. Batch loads when possible to reduce delete/load churn
    4. Defer parent deletion to avoid premature eviction
    5. Return operations list
    """
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    try:
        # Parse DAG structure
        vertices = dag.get('vertices', [])
        if not vertices:
            if 'nodes' in dag:
                vertices = dag['nodes']
            else:
                vertices = list(dag.keys()) if isinstance(dag, dict) else []
        
        parents_map = dag.get('parents', {})
        if not parents_map and isinstance(dag, dict):
            parents_map = {v: list(parents) for v, parents in dag.items()}
        
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        
        if not inputs:
            inputs = set(v for v in vertices if not parents_map.get(v, []))
        
        # Build children map
        children_map = {v: [] for v in vertices}
        for v in vertices:
            for p in parents_map.get(v, []):
                if p in children_map:
                    children_map[p].append(v)
        
        if not outputs:
            outputs = set(v for v in vertices if not children_map.get(v, []))
        
        topo_order = list(vertices)
        n = len(topo_order)
        
        # O(1) position lookup
        pos_map = {v: i for i, v in enumerate(topo_order)}
        
        # Compute last_use in O(n + E) using reverse topo order
        last_use = {}
        for v in reversed(topo_order):
            if v in outputs:
                last_use[v] = n  # Needed until the end
            elif not children_map.get(v, []):
                last_use[v] = n  # Leaf: keep until end
            else:
                max_pos = 0
                for child in children_map.get(v, []):
                    child_pos = pos_map.get(child, n)
                    if child_pos > max_pos:
                        max_pos = child_pos
                last_use[v] = max_pos
        
        # Build operation list
        ops = []
        in_fast = set()
        is_blue = set(inputs)
        
        # Count how many times each vertex is used as a parent (for prioritization)
        parent_count = {}
        for v in vertices:
            for p in parents_map.get(v, []):
                parent_count[p] = parent_count.get(p, 0) + 1
        
        # Compute next_use for each vertex (first future position where needed)
        next_use = {}
        for v in topo_order:
            pos = pos_map[v]
            if v in outputs:
                next_use[v] = n  # Needed until the end
            elif not children_map.get(v, []):
                next_use[v] = n  # Leaf: keep until end
            else:
                min_pos = n
                for child in children_map.get(v, []):
                    child_pos = pos_map.get(child, n)
                    if child_pos < min_pos:
                        min_pos = child_pos
                next_use[v] = min_pos
        
        # Process each vertex in topological order
        for i, v in enumerate(topo_order):
            if time.time() > deadline:
                break
            
            if v in inputs:
                continue
            
            parents = parents_map.get(v, [])
            
            # Determine which parents need to be loaded
            parents_to_load = []
            for p in parents:
                if p in is_blue:
                    continue
                if p in in_fast:
                    continue
                parents_to_load.append(p)
            
            # If we need to load parents and budget is full, evict first
            if parents_to_load:
                needed_slots = len(parents_to_load)
                current_slots = len(in_fast)
                if current_slots + needed_slots > fast_memory_budget:
                    # Need to free up slots
                    num_to_evict = current_slots + needed_slots - fast_memory_budget
                    # Prioritize keeping vertices with earlier next_use (more urgent)
                    # and higher parent_count (more frequently used)
                    candidates = list(in_fast)
                    # Sort by (next_use, -parent_count) - evict those with latest next_use and lowest count
                    candidates.sort(key=lambda x: (next_use.get(x, n), -parent_count.get(x, 0)))
                    # Evict from the end (latest next_use, lowest count)
                    for _ in range(num_to_evict):
                        if candidates:
                            to_delete = candidates.pop()
                            ops.append(['delete', to_delete])
                            in_fast.remove(to_delete)
            
            # Load all needed parents
            for p in parents_to_load:
                ops.append(['load', p])
                in_fast.add(p)
            
            # Compute v
            ops.append(['compute', v])
            in_fast.add(v)
            
            # If v is an output, store it to make it blue
            if v in outputs:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
            
            # Defer deletion: only delete parents if their last_use is strictly before 
            # the next position where they might be needed, AND we're over budget
            # This avoids premature eviction
            if len(in_fast) > fast_memory_budget:
                # Over budget, need to delete something
                # Delete the vertex with the latest last_use that is no longer needed soon
                candidates = list(in_fast)
                # Don't delete v if it's an output (already stored) or if it's needed
                candidates = [x for x in candidates if x != v]
                if candidates:
                    # Evict the one with latest last_use
                    to_delete = max(candidates, key=lambda x: last_use.get(x, n))
                    ops.append(['delete', to_delete])
                    in_fast.remove(to_delete)
            
            # Also clean up: delete parents that are definitely no longer needed
            # (last_use < i, meaning they won't be used again from this point)
            for p in parents:
                if p in in_fast and p != v and last_use.get(p, n) < i:
                    ops.append(['delete', p])
                    in_fast.remove(p)
        
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
        
    except Exception:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
