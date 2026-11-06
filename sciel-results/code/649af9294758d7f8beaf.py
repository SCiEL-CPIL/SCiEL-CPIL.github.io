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
    DEADLINE = 85.0  # wall-clock limit with safety margin
    
    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        output_set = set(outputs)
        input_set = set(inputs)
        
        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | input_set | output_set
        
        # Compute topological order using Kahn's algorithm
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in input_set:
            in_degree[v] = 0
        
        queue = deque([v for v in all_vertices if in_degree.get(v, 0) == 0])
        topo_order = []
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        # Map vertex to index in topo_order
        topo_index = {v: i for i, v in enumerate(topo_order)}
        n = len(topo_order)
        
        # Precompute last_use[v]: the highest topo index of any vertex that needs v as a parent
        last_use = {}
        for v in all_vertices:
            max_idx = -1
            for c in children.get(v, []):
                idx = topo_index.get(c, -1)
                if idx > max_idx:
                    max_idx = idx
            if v in output_set:
                max_idx = max(max_idx, n - 1)
            last_use[v] = max_idx
        
        # First-pass: compute the "naive" schedule to understand the baseline
        # Naive: load inputs, compute in topo order, keep everything, no deletions until end
        # This gives us a reference point
        
        # Key insight: We want to minimize transfers (load + store operations)
        # Strategy:
        # 1. Use a greedy approach that deletes vertices as soon as they're no longer needed
        # 2. Track fast memory usage and avoid unnecessary loads
        # 3. For layered DAGs, we can exploit the layer structure to minimize memory pressure
        
        # Detect if the DAG is layered (all edges go from layer i to layer i+1)
        # Assign layers based on longest path from inputs
        layer = {}
        for v in input_set:
            layer[v] = 0
        
        # Process in topo order to assign layers
        for v in topo_order:
            if v not in layer:
                ps = parents.get(v, [])
                if ps:
                    layer[v] = max(layer.get(p, 0) for p in ps) + 1
                else:
                    layer[v] = 0
        
        # Group vertices by layer
        layers = defaultdict(list)
        for v in topo_order:
            layers[layer[v]].append(v)
        
        num_layers = max(layer.values()) + 1 if layer else 0
        
        # Strategy for layered DAGs:
        # Process layer by layer. For each layer:
        # 1. Ensure all parents (from previous layer) are in fast memory
        # 2. Compute all vertices in this layer
        # 3. Delete vertices from the previous layer that are no longer needed
        
        ops = []
        fast_mem = set()
        
        # Track remaining uses for each vertex
        remaining_uses = {}
        for v in all_vertices:
            if v in output_set:
                remaining_uses[v] = 1  # Must be kept until end
            else:
                remaining_uses[v] = len(children.get(v, []))
        
        # If the DAG is layered, use layer-by-layer processing
        if num_layers > 1 and len(layers) == num_layers:
            for l in range(num_layers):
                if time.time() > DEADLINE:
                    break
                
                layer_vertices = layers[l]
                
                # First, ensure all parents of vertices in this layer are in fast memory
                for v in layer_vertices:
                    ps = parents.get(v, [])
                    for p in ps:
                        if p not in fast_mem:
                            ops.append(['load', p])
                            fast_mem.add(p)
                
                # Compute all vertices in this layer
                for v in layer_vertices:
                    ps = parents.get(v, [])
                    if not ps:
                        # Input vertex
                        if v not in fast_mem:
                            ops.append(['load', v])
                            fast_mem.add(v)
                    else:
                        ops.append(['compute', v])
                        fast_mem.add(v)
                    
                    # Decrement remaining uses for parents
                    for p in ps:
                        remaining_uses[p] -= 1
                        # Delete if no longer needed and not an output
                        if remaining_uses[p] == 0 and p not in output_set:
                            ops.append(['delete', p])
                            fast_mem.discard(p)
                
                # After processing this layer, delete vertices from previous layers that are no longer needed
                # (This is already handled above, but let's be explicit for clarity)
        
        else:
            # General DAG: process in topological order with immediate deletion
            for v in topo_order:
                if time.time() > DEADLINE:
                    break
                
                ps = parents.get(v, [])
                
                if not ps:
                    # Input vertex
                    if v not in fast_mem:
                        ops.append(['load', v])
                        fast_mem.add(v)
                else:
                    # Ensure parents are in fast memory
                    for p in ps:
                        if p not in fast_mem:
                            ops.append(['load', p])
                            fast_mem.add(p)
                    
                    ops.append(['compute', v])
                    fast_mem.add(v)
                    
                    # Decrement remaining uses for parents
                    for p in ps:
                        remaining_uses[p] -= 1
                        if remaining_uses[p] == 0 and p not in output_set:
                            ops.append(['delete', p])
                            fast_mem.discard(p)
        
        # Ensure all outputs are in fast memory at the end
        for o in outputs:
            if o not in fast_mem:
                ops.append(['load', o])
                fast_mem.add(o)
        
        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            return ops[:20000]
            
    except Exception as e:
        return []
# EVOLVE-BLOCK-END
