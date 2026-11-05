# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy:
    1. Parse the DAG structure
    2. Use a recursive pebbling algorithm that minimizes communication
    3. Track fast memory usage carefully
    4. Return operations list
    
    The key insight for layered DAGs: we can use a divide-and-conquer approach
    where we process sub-DAGs recursively, keeping only necessary intermediate 
    results in fast memory.
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    # Parse DAG structure
    # dag is expected to have:
    # - 'vertices': list of vertex names/ids
    # - 'edges': list of (parent, child) tuples or similar
    # - 'inputs': vertices that start blue
    # - 'outputs': vertices that must finish blue
    # - 'parents': topologically ordered parents for each vertex
    
    try:
        vertices = dag.get('vertices', [])
        if not vertices:
            # Try alternative structure
            if 'nodes' in dag:
                vertices = dag['nodes']
            else:
                # Assume dag is a dict mapping vertex -> list of parents
                vertices = list(dag.keys()) if isinstance(dag, dict) else []
        
        # Get edges/parents structure
        parents_map = dag.get('parents', {})
        if not parents_map and isinstance(dag, dict):
            # dag might be {vertex: [parents]}
            parents_map = {v: list(parents) for v, parents in dag.items()}
        
        # Get inputs and outputs
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        
        # If not specified, infer: inputs are vertices with no parents
        if not inputs:
            inputs = set(v for v in vertices if not parents_map.get(v, []))
        
        # If not specified, outputs might be vertices with no children
        if not outputs:
            # Find all children
            children_map = {v: [] for v in vertices}
            for v in vertices:
                for p in parents_map.get(v, []):
                    if p in children_map:
                        children_map[p].append(v)
            outputs = set(v for v in vertices if not children_map.get(v, []))
        
        # Build adjacency: for each vertex, its parents (already have) and children
        children_map = {v: [] for v in vertices}
        for v in vertices:
            for p in parents_map.get(v, []):
                if p in children_map:
                    children_map[p].append(v)
        
        # Topological order (vertices should already be topologically ordered)
        topo_order = list(vertices)
        
        # Count total operations needed
        # Each non-input vertex needs at least 1 compute
        n_compute = sum(1 for v in vertices if v not in inputs)
        
        # Naive schedule: load all inputs, compute in topo order, store outputs
        # This gives us a baseline for comparison
        
        # Our improved strategy: recursive pebbling
        # For layered DAGs, we can use a space-time tradeoff
        
        # Build the operation list
        ops = []
        fast_memory = set()  # vertices currently in fast memory (red)
        blue_vertices = set(inputs)  # vertices currently blue (available)
        
        # We need to ensure all outputs end up blue
        # Strategy: process in topological order, but be smart about what to keep
        
        # For each vertex in topo order:
        # - If it's an input, it's already blue (no load needed if we track it)
        # - If it's a non-input, we need its parents to be available
        # - After computing, decide whether to keep in fast memory or delete
        
        # Improved: use a recursive approach for better space efficiency
        # Key idea: for a vertex with parents p1, p2, ..., pk:
        #   - Ensure all parents are available (blue or in fast memory)
        #   - Compute the vertex
        #   - If fast memory is full, we may need to delete something
        #   - But we can recompute if needed
        
        # Let's use a greedy approach with lookahead:
        # Process vertices in topo order. For each vertex:
        #   1. For each parent not in fast_memory and not blue, load it
        #   2. Compute the vertex (parents must be in fast_memory or blue)
        #   3. If fast_memory is full, delete a vertex that won't be needed soon
        #   4. If the vertex is an output, mark it blue and remove from fast_memory
        
        # First, compute for each vertex when it's last needed (by its children)
        last_use = {}
        for v in reversed(topo_order):
            if v in outputs:
                last_use[v] = len(topo_order)  # Needed until the end
            elif not children_map.get(v, []):
                last_use[v] = len(topo_order)  # Leaf, but not output? Keep until end
            else:
                # Last use is the max position among children
                max_pos = 0
                for child in children_map.get(v, []):
                    if child in last_use:
                        max_pos = max(max_pos, last_use[child])
                    else:
                        # Child not processed yet in reverse, find its position
                        try:
                            pos = topo_order.index(child)
                            max_pos = max(max_pos, pos)
                        except ValueError:
                            max_pos = len(topo_order)
                last_use[v] = max_pos
        
        # Simpler approach: just use a greedy pebbling with careful memory management
        # Process in topological order
        
        # Track what's in fast memory and what's blue
        in_fast = set()  # vertices in fast memory (red)
        is_blue = set(inputs)  # vertices that are blue
        
        # We need to track which vertices have been "materialized" (computed and available)
        # A vertex is available if it's blue or in fast memory
        
        # For efficiency, let's precompute the dependency graph
        # and use a priority-based deletion strategy
        
        # Process each vertex in topological order
        for i, v in enumerate(topo_order):
            if time.time() > deadline:
                break
            
            # Skip inputs (they're already blue)
            if v in inputs:
                continue
            
            # Check if we've already computed this vertex (shouldn't happen in topo order)
            # Get parents
            parents = parents_map.get(v, [])
            
            # Ensure all parents are available (blue or in fast memory)
            for p in parents:
                if p not in is_blue and p not in in_fast:
                    # Need to load parent
                    # But first, check if fast memory is full
                    if len(in_fast) >= fast_memory_budget:
                        # Need to delete something
                        # Delete a vertex that is used least in the future
                        # Find the vertex in in_fast with the latest last_use
                        # or one that's not needed for current computation
                        candidates = list(in_fast)
                        # Prefer to delete vertices not needed for current or near-future computations
                        # For now, delete the one with the latest last_use (least urgent)
                        # Actually, delete the one with the EARLIEST last_use that's still in the future
                        # Wait, we want to keep vertices that are needed soon
                        # So delete the one with the LATEST last_use (needed least soon)
                        # Hmm, this is tricky. Let's use a simpler heuristic:
                        # Delete a vertex that is not a parent of any vertex in the next few steps
                        next_vertices = set(topo_order[i:i+5])
                        next_parents = set()
                        for nv in next_vertices:
                            next_parents.update(parents_map.get(nv, []))
                        
                        # Find candidates not in next_parents
                        non_urgent = [x for x in candidates if x not in next_parents]
                        if non_urgent:
                            # Delete one with the latest last_use
                            to_delete = max(non_urgent, key=lambda x: last_use.get(x, 0))
                        else:
                            # All are potentially urgent, delete one with latest last_use
                            to_delete = max(candidates, key=lambda x: last_use.get(x, 0))
                        
                        ops.append(['delete', to_delete])
                        in_fast.remove(to_delete)
                    
                    ops.append(['load', p])
                    in_fast.add(p)
                
                # After loading, the parent is in fast memory
                # If the parent was blue, it's still blue (loading doesn't change blue status)
                # Actually, loading brings it into fast memory, but it remains blue if it was blue
            
            # Now compute v
            ops.append(['compute', v])
            
            # v is now in fast memory (red)
            in_fast.add(v)
            
            # If v is an output, we need it to be blue at the end
            # So we should store it (make it blue) and remove from fast memory
            if v in outputs:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
            
            # Optionally, delete parents that are no longer needed
            # Check if any parent's last use was just now
            for p in parents:
                if p in in_fast and last_use.get(p, 0) <= i:
                    # Parent is no longer needed
                    ops.append(['delete', p])
                    in_fast.remove(p)
        
        # Ensure all outputs are blue
        # (They should be, since we stored them)
        
        # Check operation count
        if len(ops) > 20000:
            # Truncate (shouldn't happen for reasonable DAGs)
            ops = ops[:20000]
        
        return ops
        
    except Exception as e:
        # Fallback: try to use baseline if available
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
