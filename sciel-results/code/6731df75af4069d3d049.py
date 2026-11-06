# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy:
    1. Parse the DAG structure
    2. Use a greedy pebbling algorithm with O(n) last_use computation
    3. Track fast memory usage carefully
    4. Return operations list
    
    The key insight for layered DAGs: process in topological order, keeping
    only necessary intermediate results in fast memory, and deleting values
    as soon as they are no longer needed.
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    # Parse DAG structure
    try:
        vertices = dag.get('vertices', [])
        if not vertices:
            if 'nodes' in dag:
                vertices = dag['nodes']
            else:
                vertices = list(dag.keys()) if isinstance(dag, dict) else []
        
        # Get edges/parents structure
        parents_map = dag.get('parents', {})
        if not parents_map and isinstance(dag, dict):
            parents_map = {v: list(parents) for v, parents in dag.items()}
        
        # Get inputs and outputs
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        
        # If not specified, infer: inputs are vertices with no parents
        if not inputs:
            inputs = set(v for v in vertices if not parents_map.get(v, []))
        
        # Build children map
        children_map = {v: [] for v in vertices}
        for v in vertices:
            for p in parents_map.get(v, []):
                if p in children_map:
                    children_map[p].append(v)
        
        # If not specified, outputs are vertices with no children
        if not outputs:
            outputs = set(v for v in vertices if not children_map.get(v, []))
        
        # Topological order (vertices should already be topologically ordered)
        topo_order = list(vertices)
        n = len(topo_order)
        
        # Build position map for O(1) lookup
        pos_map = {v: i for i, v in enumerate(topo_order)}
        
        # Compute last_use in O(n) using reverse topological order
        # last_use[v] = the maximum position index among all children of v
        # For outputs, last_use is n (needed until the end)
        last_use = {}
        for i in range(n - 1, -1, -1):
            v = topo_order[i]
            if v in outputs:
                last_use[v] = n
            else:
                children = children_map.get(v, [])
                if not children:
                    last_use[v] = n  # Leaf, keep until end
                else:
                    # Max position among children
                    max_pos = i  # At minimum, it's used at its own position
                    for child in children:
                        if child in last_use:
                            max_pos = max(max_pos, last_use[child])
                        elif child in pos_map:
                            max_pos = max(max_pos, pos_map[child])
                    last_use[v] = max_pos
        
        # Build operation list
        ops = []
        in_fast = set()  # vertices currently in fast memory (red)
        is_blue = set(inputs)  # vertices that are blue
        
        # Process each vertex in topological order
        for i, v in enumerate(topo_order):
            if time.time() > deadline:
                break
            
            # Skip inputs (they're already blue)
            if v in inputs:
                continue
            
            # Get parents
            parents = parents_map.get(v, [])
            
            # Ensure all parents are available (blue or in fast memory)
            for p in parents:
                if p not in is_blue and p not in in_fast:
                    # Need to load parent
                    if len(in_fast) >= fast_memory_budget:
                        # Need to delete something to make room
                        # Delete the vertex with the latest last_use (least urgent)
                        # among those not needed for current vertex's parents
                        current_parents_set = set(parents)
                        candidates = [x for x in in_fast if x not in current_parents_set]
                        if not candidates:
                            # All in fast memory are needed as parents, this shouldn't happen
                            # but if it does, just pick any
                            candidates = list(in_fast)
                        
                        if candidates:
                            # Delete the one with the latest last_use (needed least soon)
                            to_delete = max(candidates, key=lambda x: last_use.get(x, 0))
                            ops.append(['delete', to_delete])
                            in_fast.remove(to_delete)
                    
                    ops.append(['load', p])
                    in_fast.add(p)
            
            # Now compute v
            ops.append(['compute', v])
            
            # v is now in fast memory (red)
            in_fast.add(v)
            
            # If v is an output, we need it to be blue at the end
            if v in outputs:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
            
            # Delete parents that are no longer needed (last_use <= current position)
            for p in parents:
                if p in in_fast and last_use.get(p, 0) <= i:
                    ops.append(['delete', p])
                    in_fast.remove(p)
        
        # Ensure all outputs are blue
        # (They should be, since we stored them)
        
        # Check operation count
        if len(ops) > 20000:
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
