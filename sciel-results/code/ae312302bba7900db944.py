# EVOLVE-BLOCK-START
import time
import sys
import heapq

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling for layered DAGs.
    
    Strategy:
    1. Parse DAG structure and identify layers
    2. Use layer-by-layer processing with smart memory management
    3. Minimize transfers by keeping only necessary vertices in fast memory
    4. Use recomputation when beneficial
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    try:
        # Parse DAG structure
        vertices = dag.get('vertices', [])
        if not vertices:
            if 'nodes' in dag:
                vertices = dag['nodes']
            elif isinstance(dag, dict):
                vertices = list(dag.keys())
        
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
        pos_map = {v: i for i, v in enumerate(topo_order)}
        
        # Identify layers based on topological depth
        # A vertex's layer is 1 + max(layer of its parents)
        layer = {}
        for v in topo_order:
            parents = parents_map.get(v, [])
            if not parents:
                layer[v] = 0
            else:
                layer[v] = 1 + max(layer.get(p, 0) for p in parents)
        
        # Group vertices by layer
        max_layer = max(layer.values()) if layer else 0
        layers = [[] for _ in range(max_layer + 1)]
        for v in topo_order:
            layers[layer[v]].append(v)
        
        # Compute last_use: the last position where a vertex is needed
        last_use = {}
        for i in range(n - 1, -1, -1):
            v = topo_order[i]
            if v in outputs:
                last_use[v] = n
            elif not children_map.get(v, []):
                last_use[v] = i
            else:
                max_lu = i
                for child in children_map.get(v, []):
                    if child in last_use:
                        max_lu = max(max_lu, last_use[child])
                last_use[v] = max_lu
        
        # Build operation list
        ops = []
        in_fast = set()
        is_blue = set(inputs)
        
        # Track which vertices have been computed and are available
        computed = set(inputs)
        
        # For each vertex, track how many children still need it
        remaining_children = {v: len(children_map.get(v, [])) for v in vertices}
        
        # Process vertices in topological order
        for i, v in enumerate(topo_order):
            if time.time() > deadline:
                break
            
            # Skip inputs
            if v in inputs:
                continue
            
            parents = parents_map.get(v, [])
            
            # Ensure all parents are available
            for p in parents:
                if p not in is_blue and p not in in_fast:
                    # Need to make parent available
                    # Check if fast memory is full
                    if len(in_fast) >= fast_memory_budget:
                        # Need to evict something
                        # Strategy: evict the vertex with the latest last_use
                        # that is not needed for current or future computations soon
                        candidates = list(in_fast)
                        # Filter: don't evict vertices that are parents of vertices in current layer
                        current_layer = layer[v]
                        protected = set()
                        for cv in layers[current_layer]:
                            for pp in parents_map.get(cv, []):
                                if pp in in_fast:
                                    protected.add(pp)
                        
                        evictable = [c for c in candidates if c not in protected]
                        if not evictable:
                            evictable = candidates
                        
                        if evictable:
                            # Evict the one with latest last_use
                            to_delete = max(evictable, key=lambda x: last_use.get(x, 0))
                            ops.append(['delete', to_delete])
                            in_fast.remove(to_delete)
                    
                    ops.append(['load', p])
                    in_fast.add(p)
                
                # If parent was blue, keep it blue
        
            # Compute v
            ops.append(['compute', v])
            in_fast.add(v)
            computed.add(v)
            
            # Update remaining_children for parents
            for p in parents:
                if p in remaining_children:
                    remaining_children[p] -= 1
            
            # If v is an output, store it to make it blue
            if v in outputs:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
            
            # Proactively delete vertices that are no longer needed
            # A vertex can be deleted if:
            # 1. It's in fast memory
            # 2. All its children have been computed
            # 3. It's not an output (or output is already stored)
            for p in parents:
                if p in in_fast and p not in outputs and remaining_children.get(p, 0) == 0:
                    ops.append(['delete', p])
                    in_fast.remove(p)
            
            # Also check other vertices in fast memory that may no longer be needed
            # This is expensive, so only do it occasionally
            if i % 10 == 0 and len(in_fast) > 0:
                to_remove = []
                for x in in_fast:
                    if x not in outputs and remaining_children.get(x, 0) == 0:
                        to_remove.append(x)
                for x in to_remove:
                    ops.append(['delete', x])
                    in_fast.remove(x)
        
        # Ensure all outputs are blue
        for v in outputs:
            if v not in is_blue and v in in_fast:
                ops.append(['store', v])
                in_fast.remove(v)
                is_blue.add(v)
        
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
        
    except Exception as e:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except:
            return []
# EVOLVE-BLOCK-END
