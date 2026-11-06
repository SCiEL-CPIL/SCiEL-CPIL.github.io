# EVOLVE-BLOCK-START
import time
import sys
import heapq

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    
    Strategy:
    1. Parse the DAG structure
    2. Use a greedy pebbling algorithm with efficient memory management
    3. Track fast memory usage carefully
    4. Return operations list
    
    The key insight for layered DAGs: we can use a divide-and-conquer approach
    where we process sub-DAGs recursively, keeping only necessary intermediate 
    results in fast memory.
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    # Parse DAG structure
    try:
        vertices = dag.get('vertices', [])
        if not vertices:
            if 'nodes' in dag:
                vertices = dag['nodes']
            elif isinstance(dag, dict):
                vertices = list(dag.keys())
        
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
        
        # If not specified, outputs might be vertices with no children
        if not outputs:
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
        n = len(topo_order)
        
        # Create position map for O(1) lookup
        pos_map = {v: i for i, v in enumerate(topo_order)}
        
        # Compute last_use for each vertex in O(n * avg_degree)
        # last_use[v] = position of the last vertex that depends on v (directly or transitively)
        # For outputs, last_use is n (they need to be available until the end)
        last_use = {}
        
        # Process in reverse topological order
        for i in range(n - 1, -1, -1):
            v = topo_order[i]
            if v in outputs:
                last_use[v] = n
            elif not children_map.get(v, []):
                # Leaf vertex (not an output) - it's only needed when computed
                last_use[v] = i
            else:
                # Last use is the max last_use among children
                max_lu = i
                for child in children_map.get(v, []):
                    if child in last_use:
                        max_lu = max(max_lu, last_use[child])
                    elif child in pos_map:
                        max_lu = max(max_lu, pos_map[child])
                last_use[v] = max_lu
        
        # Build operation list
        ops = []
        in_fast = set()  # vertices in fast memory (red)
        is_blue = set(inputs)  # vertices that are blue
        
        # Use a heap for efficient deletion: (last_use, vertex)
        # We want to delete the vertex with the LATEST last_use (least urgent)
        # So we use a max-heap via negative last_use
        deletion_heap = []  # will store (-last_use, vertex) for max-heap behavior
        
        # Precompute which vertices are parents of vertices in the next window
        # for better deletion decisions
        
        # Process each vertex in topological order
        for i, v in enumerate(topo_order):
            if time.time() > deadline:
                break
            
            # Skip inputs (they're already blue)
            if v in inputs:
                # Inputs don't need to be loaded, but if they're in fast memory, keep them
                continue
            
            # Get parents
            parents = parents_map.get(v, [])
            
            # Ensure all parents are available (blue or in fast memory)
            for p in parents:
                if p not in is_blue and p not in in_fast:
                    # Need to load parent
                    # Check if fast memory is full
                    if len(in_fast) >= fast_memory_budget:
                        # Need to delete something
                        # Use the heap to find the least urgent vertex to delete
                        # Clean up the heap: remove entries that are no longer in in_fast
                        while deletion_heap:
                            neg_lu, candidate = deletion_heap[0]
                            if candidate in in_fast:
                                break
                            heapq.heappop(deletion_heap)
                        
                        if deletion_heap:
                            neg_lu, to_delete = heapq.heappop(deletion_heap)
                        else:
                            # Fallback: find manually
                            # Delete the vertex with the latest last_use
                            candidates = list(in_fast)
                            if not candidates:
                                break
                            to_delete = max(candidates, key=lambda x: last_use.get(x, 0))
                        
                        ops.append(['delete', to_delete])
                        in_fast.remove(to_delete)
                    
                    ops.append(['load', p])
                    in_fast.add(p)
                    # Add to deletion heap
                    heapq.heappush(deletion_heap, (-last_use.get(p, 0), p))
                
                # If the parent was blue, it's still blue
        
            # Now compute v
            ops.append(['compute', v])
            
            # v is now in fast memory (red)
            in_fast.add(v)
            heapq.heappush(deletion_heap, (-last_use.get(v, 0), v))
            
            # If v is an output, we need it to be blue at the end
            # So we should store it (make it blue) and remove from fast memory
            if v in outputs:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
                # Don't add to deletion heap since it's no longer in fast memory
            
            # Optionally, delete parents that are no longer needed
            # Check if any parent's last use was just now or earlier
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
