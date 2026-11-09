"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

_DEADLINE = 55.0  # Safety margin before 60s limit

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

def _pebble_improved_greedy(dag, fast_memory_budget, deadline):
    """
    Improved greedy pebbling with better eviction strategy.
    Uses remaining-use-count and layer-based heuristics to minimize transfers.
    """
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    topo = _topological_order(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    output_set = set(outputs)
    input_set = set(inputs)
    
    # Compute children
    children = defaultdict(list)
    for v in topo:
        parents = _get_parents(dag, v)
        for p in parents:
            children[p].append(v)
    
    # Compute remaining uses: for each vertex, how many children haven't been processed yet
    remaining_uses = {v: len(children[v]) for v in topo}
    
    # Compute layer for tie-breaking
    layer = _compute_layers(dag, topo)
    
    # Precompute next-use index for each vertex (when it's last needed)
    next_use = {}
    for i, v in enumerate(topo):
        if children[v]:
            # Last child in topo order
            last_child = max(children[v], key=lambda c: topo.index(c))
            next_use[v] = topo.index(last_child)
        else:
            next_use[v] = i
    
    operations = []
    in_fast = set()
    
    def evict_one():
        if not in_fast:
            return
        # Find the best vertex to evict
        # Priority:
        # 1. Non-outputs preferred for eviction
        # 2. Vertices with 0 remaining uses (done)
        # 3. Later next_use (used far in future)
        # 4. Lower remaining uses
        # 5. Older (smaller layer)
        
        candidates = list(in_fast)
        
        def sort_key(v):
            is_output = 1 if v in output_set else 0
            is_input = 1 if v in input_set else 0
            ru = remaining_uses.get(v, 0)
            nu = next_use.get(v, 0)
            lv = layer.get(v, 0)
            # We want to evict: non-output, non-input, 0 remaining uses, later next use
            # Lower key = better to evict
            return (is_output, is_input, ru, -nu, lv)
        
        candidates.sort(key=sort_key)
        v_to_evict = candidates[0]
        operations.append(['delete', v_to_evict])
        in_fast.remove(v_to_evict)
    
    # Process vertices in topological order
    for v in topo:
        if time.time() - deadline[0] > _DEADLINE:
            break
        
        parents = _get_parents(dag, v)
        
        # Skip inputs (they don't need computation)
        if not parents:
            continue
        
        # Ensure all parents are in fast memory
        for p in parents:
            if p not in in_fast:
                # Need to load p
                if len(in_fast) >= fast_memory_budget:
                    evict_one()
                operations.append(['load', p])
                in_fast.add(p)
        
        # Make room for v if needed
        if len(in_fast) >= fast_memory_budget:
            evict_one()
        
        # Compute v
        operations.append(['compute', v])
        in_fast.add(v)
        
        # Update remaining uses for parents
        for p in parents:
            if p in remaining_uses:
                remaining_uses[p] -= 1
        
        # Immediately delete v if it has no more uses and is not an output
        if remaining_uses.get(v, 0) == 0 and v not in output_set:
            operations.append(['delete', v])
            in_fast.remove(v)
    
    # Phase 2: Store all outputs
    for o in outputs:
        if time.time() - deadline[0] > _DEADLINE:
            break
        if o not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                evict_one()
            operations.append(['load', o])
            in_fast.add(o)
        
        operations.append(['store', o])
        in_fast.remove(o)
    
    return operations

def pebble(dag, fast_memory_budget):
    deadline = [time.time()]
    
    try:
        # Use improved greedy pebbling
        ops = _pebble_improved_greedy(dag, fast_memory_budget, deadline)
        
        if not ops:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        
        if _validate_operations(ops, dag, fast_memory_budget):
            if len(ops) <= 20000:
                return ops
        
    except Exception:
        pass
    
    # Final fallback to baseline
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
