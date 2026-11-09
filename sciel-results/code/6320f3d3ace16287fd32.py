"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

_DEADLINE = 55.0  # Safety margin before 60s limit

def _compute_future_usage(dag, topo, layer):
    """
    Compute for each vertex how many times it will be needed in future layers.
    Returns dict: vertex -> (total_children, remaining_children)
    """
    children = defaultdict(list)
    for v in topo:
        parents = _get_parents(dag, v)
        for p in parents:
            children[p].append(v)
    
    total_children = {v: len(children[v]) for v in topo}
    remaining_children = {v: len(children[v]) for v in topo}
    
    # Process in reverse topological order to count remaining uses
    for v in reversed(topo):
        for p in _get_parents(dag, v):
            remaining_children[p] = remaining_children.get(p, 0)  # Already counted
            # Actually, we need to track which children have been processed
            pass
    
    # Simpler approach: for each vertex, count how many of its children are in higher layers
    future_usage = {}
    for v in topo:
        count = 0
        for c in children[v]:
            if layer[c] > layer[v]:
                count += 1
        future_usage[v] = count
    
    return future_usage, children

def _get_parents(dag, vertex):
    """Get parent vertices of a given vertex."""
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        return parents.get(vertex, [])
    elif hasattr(dag, 'parents'):
        return dag.parents.get(vertex, []) if isinstance(dag.parents, dict) else list(dag.parents[vertex])
    else:
        return []

def _get_outputs(dag):
    """Get output vertices of the DAG."""
    if isinstance(dag, dict):
        outputs = dag.get('outputs', [])
        return list(outputs)
    elif hasattr(dag, 'outputs'):
        return list(dag.outputs)
    else:
        return []

def _get_all_vertices(dag):
    """Get all vertices in the DAG."""
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        if vertices is not None:
            return list(vertices)
        # Try to infer from parents
        parents = dag.get('parents', {})
        all_v = set()
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        return list(all_v)
    elif hasattr(dag, 'vertices'):
        return list(dag.vertices)
    else:
        return []

def _get_inputs(dag):
    """Get input vertices (those with no parents)."""
    all_v = _get_all_vertices(dag)
    inputs = []
    for v in all_v:
        if not _get_parents(dag, v):
            inputs.append(v)
    return inputs

def _topological_order(dag):
    """Get topological order of vertices."""
    if isinstance(dag, dict) and 'topo_order' in dag:
        return list(dag['topo_order'])
    
    all_v = _get_all_vertices(dag)
    parents = {}
    children = defaultdict(list)
    
    for v in all_v:
        ps = _get_parents(dag, v)
        parents[v] = ps
        for p in ps:
            children[p].append(v)
    
    # Kahn's algorithm
    in_degree = {v: len(parents[v]) for v in all_v}
    queue = [v for v in all_v if in_degree[v] == 0]
    topo = []
    while queue:
        v = queue.pop(0)
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo) != len(all_v):
        # Cycle or error, just return all vertices
        return all_v
    
    return topo

def _compute_layers(dag, topo):
    """Compute the layer (depth) of each vertex."""
    layer = {}
    for v in topo:
        parents = _get_parents(dag, v)
        if not parents:
            layer[v] = 0
        else:
            layer[v] = max(layer[p] for p in parents) + 1
    return layer

def _validate_operations(operations, dag, fast_memory_budget):
    """Validate that operations are legal."""
    in_fast = set()
    for op, v in operations:
        if op == 'load':
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        elif op == 'compute':
            parents = _get_parents(dag, v)
            for p in parents:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        else:
            return False
    return True

def _count_transfers(operations):
    """Count load and store operations."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def _naive_schedule_transfers(dag, fast_memory_budget):
    """Estimate naive schedule transfers."""
    all_v = _get_all_vertices(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    
    # Naive: load all inputs, compute all, store all outputs
    # This is an estimate
    transfers = len(inputs) + len(outputs)
    return transfers

def _pebble_lru_layered(dag, fast_memory_budget, deadline):
    """
    Pebbling strategy using layered processing and smart eviction based on future usage.
    """
    start_time = time.time()
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    topo = _topological_order(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    output_set = set(outputs)
    
    # Compute layers
    layer = _compute_layers(dag, topo)
    
    # Group vertices by layer
    layers = defaultdict(list)
    for v in topo:
        layers[layer[v]].append(v)
    
    max_layer = max(layers.keys()) if layers else 0
    
    # Pre-compute children and future usage
    children = defaultdict(list)
    for v in topo:
        parents = _get_parents(dag, v)
        for p in parents:
            children[p].append(v)
    
    # For each vertex, compute how many of its children are in higher layers
    # This tells us how many more times we'll need this vertex
    future_need = {}
    for v in topo:
        count = 0
        for c in children[v]:
            if layer[c] > layer[v]:
                count += 1
        future_need[v] = count
    
    operations = []
    in_fast = set()
    
    # Track which vertices have been fully processed (all children done)
    remaining_children = {v: len(children[v]) for v in topo}
    
    def evict_one():
        if not in_fast:
            return
        # Find the best vertex to evict based on:
        # 1. Not an output (outputs must be stored at end)
        # 2. Lower future_need (fewer remaining uses)
        # 3. Smaller layer (older)
        # 4. Already has all children processed (remaining_children == 0)
        
        candidates = list(in_fast)
        
        def sort_key(v):
            # We want to evict the 'least valuable' vertex.
            # Score: Lower tuple = better to evict
            
            # 1. Outputs should be kept (high penalty)
            is_output = 1 if v in output_set else 0
            
            # 2. Future need: fewer remaining uses = better to evict
            fn = future_need.get(v, 0)
            
            # 3. Layer: smaller layer = better to evict (older)
            lv = layer.get(v, 0)
            
            # 4. Remaining children: 0 = better to evict (done)
            rc = remaining_children.get(v, 0)
            
            # Combine: prioritize non-outputs with low future need and no remaining children
            # If remaining_children == 0, this vertex won't be needed again
            if rc == 0:
                # This vertex is done, good candidate for eviction
                return (is_output, 0, lv)  # Low future need score
            else:
                # This vertex still needed, less ideal to evict
                return (is_output, fn, lv)
            
        # Sort by key, evict the first one (lowest score)
        candidates.sort(key=sort_key)
        v_to_evict = candidates[0]
        operations.append(['delete', v_to_evict])
        in_fast.remove(v_to_evict)

    # Process each layer
    for l in range(max_layer + 1):
        if time.time() - deadline[0] > _DEADLINE:
            break
            
        # Vertices in this layer
        current_layer_verts = layers.get(l, [])
        
        for v in current_layer_verts:
            if time.time() - deadline[0] > _DEADLINE:
                break
                
            parents = _get_parents(dag, v)
            
            # Ensure all parents are in fast memory
            for p in parents:
                if p not in in_fast:
                    # Need to load p
                    # Check if fast memory is full
                    if len(in_fast) >= fast_memory_budget:
                        evict_one()
                    
                    operations.append(['load', p])
                    in_fast.add(p)
            
            # Compute v
            # Check if fast memory is full (need space for v)
            if len(in_fast) >= fast_memory_budget:
                evict_one()
            
            operations.append(['compute', v])
            in_fast.add(v)
            
            # Update remaining_children for parents
            for p in parents:
                if p in remaining_children:
                    remaining_children[p] -= 1

    # Phase 2: Ensure all outputs are in fast memory
    for o in outputs:
        if time.time() - deadline[0] > _DEADLINE:
            break
        if o not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                evict_one()
            operations.append(['load', o])
            in_fast.add(o)
        
        # Store the output
        operations.append(['store', o])
        in_fast.remove(o)
        
    return operations

def pebble(dag, fast_memory_budget):
    deadline = [time.time()]
    
    try:
        ops = _pebble_lru_layered(dag, fast_memory_budget, deadline)
        
        if not ops:
            # Fallback to baseline if no ops generated
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        
        # Validate
        if _validate_operations(ops, dag, fast_memory_budget):
            if len(ops) <= 20000:
                return ops
        
        # If invalid, try simpler approach with better eviction
        ops_simple = []
        all_v = _get_all_vertices(dag)
        topo = _topological_order(dag)
        inputs = _get_inputs(dag)
        outputs = _get_outputs(dag)
        output_set = set(outputs)
        
        # Pre-compute layers and future need for better eviction
        layer = _compute_layers(dag, topo)
        children = defaultdict(list)
        for v in topo:
            parents = _get_parents(dag, v)
            for p in parents:
                children[p].append(v)
        
        future_need = {}
        for v in topo:
            count = sum(1 for c in children[v] if layer[c] > layer[v])
            future_need[v] = count
        
        remaining_children = {v: len(children[v]) for v in topo}
        in_fast = set()
        
        def evict_simple():
            if not in_fast:
                return
            candidates = list(in_fast)
            def sort_key(v):
                is_output = 1 if v in output_set else 0
                fn = future_need.get(v, 0)
                rc = remaining_children.get(v, 0)
                if rc == 0:
                    return (is_output, 0, layer.get(v, 0))
                else:
                    return (is_output, fn, layer.get(v, 0))
            candidates.sort(key=sort_key)
            v_to_evict = candidates[0]
            ops_simple.append(['delete', v_to_evict])
            in_fast.remove(v_to_evict)
        
        for v in topo:
            if time.time() - deadline[0] > _DEADLINE:
                break
            parents = _get_parents(dag, v)
            if not parents:
                continue
            for p in parents:
                if p not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        evict_simple()
                    ops_simple.append(['load', p])
                    in_fast.add(p)
            if len(in_fast) >= fast_memory_budget:
                evict_simple()
            ops_simple.append(['compute', v])
            in_fast.add(v)
            for p in parents:
                if p in remaining_children:
                    remaining_children[p] -= 1
        
        for o in outputs:
            if o not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    evict_simple()
                ops_simple.append(['load', o])
                in_fast.add(o)
            ops_simple.append(['store', o])
            in_fast.remove(o)
        
        if _validate_operations(ops_simple, dag, fast_memory_budget):
            if len(ops_simple) <= 20000:
                return ops_simple
                
    except Exception:
        pass
    
    # Final fallback to baseline
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
