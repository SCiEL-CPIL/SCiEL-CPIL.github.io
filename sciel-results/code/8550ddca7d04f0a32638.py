# EVOLVE-BLOCK-START
import time
import heapq

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling with min-cut-inspired eviction.
    
    Strategy:
    1. Parse the DAG structure
    2. Compute future reuse costs for each vertex
    3. Use min-cut-inspired cost model for eviction decisions
    4. Process vertices in topological order with optimal eviction
    """
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s safety margin
    
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
        
        # Compute future reuse count: how many times each vertex will be needed
        # as a parent in future computations
        future_reuse = {}
        for v in topo_order:
            pos = pos_map[v]
            count = 0
            for child in children_map.get(v, []):
                child_pos = pos_map.get(child, n)
                if child_pos > pos:
                    count += 1
            future_reuse[v] = count
        
        # Compute last_use position for each vertex
        last_use = {}
        for v in reversed(topo_order):
            if v in outputs:
                last_use[v] = n
            elif not children_map.get(v, []):
                last_use[v] = n
            else:
                max_pos = 0
                for child in children_map.get(v, []):
                    child_pos = pos_map.get(child, n)
                    if child_pos > max_pos:
                        max_pos = child_pos
                last_use[v] = max_pos
        
        # Compute next_use position for each vertex
        next_use = {}
        for v in topo_order:
            pos = pos_map[v]
            if v in outputs:
                next_use[v] = n
            elif not children_map.get(v, []):
                next_use[v] = n
            else:
                min_pos = n
                for child in children_map.get(v, []):
                    child_pos = pos_map.get(child, n)
                    if child_pos < min_pos:
                        min_pos = child_pos
                next_use[v] = min_pos
        
        # Build operation list
        ops = []
        in_fast = set()
        is_blue = set(inputs)
        
        # Min-cut inspired cost model:
        # Cost of keeping a vertex in fast memory = 1 (occupies a slot)
        # Cost of evicting a vertex = future_reuse[v] (number of future loads needed)
        # Net benefit of keeping = future_reuse[v] - 1
        # Evict vertices with lowest net benefit (or negative benefit)
        
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
                    
                    # Min-cut inspired eviction:
                    # Score each candidate by (future_reuse - urgency_factor)
                    # Vertices with low future_reuse and high next_use are best to evict
                    candidates = list(in_fast)
                    
                    # Compute eviction score: lower score = better to evict
                    # Score = -future_reuse + (next_use / n) * 0.5
                    # We want to keep vertices with high future_reuse and low next_use
                    def eviction_score(x):
                        fr = future_reuse.get(x, 0)
                        nu = next_use.get(x, n)
                        # Normalize next_use to [0, 1]
                        nu_norm = nu / max(n, 1)
                        # Score: lower is better to evict
                        # High future_reuse -> high score (keep)
                        # Low next_use (soon needed) -> high score (keep)
                        return fr * 1.0 + (1.0 - nu_norm) * 0.5
                    
                    # Sort by eviction score ascending (evict lowest scores first)
                    candidates.sort(key=eviction_score)
                    
                    # Evict from the start (lowest scores)
                    for _ in range(num_to_evict):
                        if candidates:
                            to_delete = candidates.pop(0)
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
            
            # If over budget, evict using min-cut inspired scoring
            if len(in_fast) > fast_memory_budget:
                candidates = list(in_fast)
                candidates = [x for x in candidates if x != v]
                if candidates:
                    # Use same eviction score
                    def eviction_score2(x):
                        fr = future_reuse.get(x, 0)
                        nu = next_use.get(x, n)
                        nu_norm = nu / max(n, 1)
                        return fr * 1.0 + (1.0 - nu_norm) * 0.5
                    
                    candidates.sort(key=eviction_score2)
                    to_delete = candidates[0]
                    ops.append(['delete', to_delete])
                    in_fast.remove(to_delete)
            
            # Clean up: delete parents that are definitely no longer needed
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
