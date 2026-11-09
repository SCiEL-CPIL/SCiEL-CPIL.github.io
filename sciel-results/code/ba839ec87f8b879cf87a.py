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
    queue = deque([v for v in all_v if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo) != len(all_v):
        # Cycle or error, just return all vertices
        return all_v
    
    return topo

def _compute_layers(dag):
    """Compute topological layers (depth) for each vertex."""
    all_v = _get_all_vertices(dag)
    if not all_v:
        return {}
    
    parents = {v: _get_parents(dag, v) for v in all_v}
    depth = {}
    queue = deque()
    
    # Initialize inputs
    for v in all_v:
        if not parents[v]:
            depth[v] = 0
            queue.append(v)
            
    while queue:
        v = queue.popleft()
        # Find children
        for u in all_v:
            if v in parents[u]:
                if u not in depth:
                    depth[u] = depth[v] + 1
                    queue.append(u)
                else:
                    depth[u] = max(depth[u], depth[v] + 1)
                    
    return depth

def _pebble_layered(dag, fast_memory_budget, deadline):
    """
    Layered pebbling strategy.
    Works well if max layer width <= fast_memory_budget.
    """
    start_time = time.time()
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
        
    depths = _compute_layers(dag)
    if not depths:
        return []
        
    # Group vertices by layer
    layers = defaultdict(list)
    for v, d in depths.items():
        layers[d].append(v)
        
    max_layer = max(layers.keys()) if layers else -1
    
    # Check if any layer exceeds budget. If so, this strategy might fail or be suboptimal.
    # We will still try, but we might need to split layers or fall back.
    # For now, assume we can handle it by careful eviction or just fail validation.
    
    operations = []
    in_fast = set()
    
    # Track which parents are still needed for future layers
    # remaining_children[parent] = count of children in current/next layers not yet computed
    remaining_children = defaultdict(int)
    for v in all_v:
        for p in _get_parents(dag, v):
            remaining_children[p] += 1
            
    # Precompute children for quick access
    children = defaultdict(list)
    for v in all_v:
        for p in _get_parents(dag, v):
            children[p].append(v)
            
    outputs = set(_get_outputs(dag))
    
    # Process Layer 0 (Inputs)
    if 0 in layers:
        for v in layers[0]:
            if len(in_fast) >= fast_memory_budget:
                # Should not happen if width <= budget
                # Evict someone? We can't evict inputs if we need them.
                # If width > budget, this strategy is invalid.
                return None 
            operations.append(['load', v])
            in_fast.add(v)
            
    # Process subsequent layers
    for l in range(1, max_layer + 1):
        if time.time() - deadline[0] > _DEADLINE:
            break
            
        if l not in layers:
            continue
            
        # Process vertices in this layer
        # We need parents from layer l-1 to be in fast memory.
        # They should be, unless we evicted them.
        
        for v in layers[l]:
            parents = _get_parents(dag, v)
            
            # Ensure parents are in fast memory
            for p in parents:
                if p not in in_fast:
                    # This means we evicted a parent prematurely or it was an input not loaded?
                    # In a strict layered graph, parents of layer l are in layer l-1.
                    # If we didn't evict them, they are there.
                    # If we did, we need to reload.
                    # Reloading an input is fine. Reloading a computed value requires it to be in slow memory (stored).
                    # We only store outputs. We do NOT store intermediate values in this strategy.
                    # So if a parent was evicted, we CANNOT reload it unless we recompute it.
                    # This strategy assumes we DON'T evict parents until all their children are computed.
                    # So if p is not in in_fast, it's an error in our logic or budget exceeded.
                    return None
                    
            # Compute v
            if len(in_fast) >= fast_memory_budget:
                # Can we delete something?
                # We can delete parents that have no more children in this layer or future layers.
                # But we are computing v, so we need space for v.
                # We should have deleted unneeded parents BEFORE computing v.
                
                # Let's try to delete unneeded parents from previous layer
                deleted = False
                for p in parents:
                    # If p has no other children in this layer or future layers, we can delete it AFTER computing v?
                    # No, we need space for v.
                    # We can delete a parent q from layer l-1 if ALL its children have been computed.
                    # We are processing layer l sequentially.
                    # If we delete q, we must ensure q is not a parent of any UNPROCESSED vertex in layer l.
                    
                    # This is complex. Simpler: Check if we have space.
                    # If not, find a vertex in in_fast that is NOT a parent of v and is "dead" (no future uses).
                    # A vertex u in layer l-1 is dead if all its children (in layer l) are processed.
                    
                    # We track processed vertices in current layer?
                    # Let's just check if we can delete any vertex from layer l-1 that is fully used.
                    
                    # To keep it simple and robust:
                    # If we are out of space, this strategy fails for this budget.
                    return None
                    
            operations.append(['compute', v])
            in_fast.add(v)
            
            # If v is an output, store it
            if v in outputs:
                operations.append(['store', v])
                in_fast.remove(v)
                
        # After processing all vertices in layer l, clean up layer l-1
        if l - 1 in layers:
            for p in layers[l-1]:
                # If p is still in fast memory, check if it has any unprocessed children
                # All children of p are in layer l (assuming strict layering) or later.
                # If strict layering, children are only in l.
                # So if we finished layer l, p has no more children.
                if p in in_fast:
                    # Check if p is an output? We stored it if it was.
                    # If it's not an output, we can delete it.
                    # If it IS an output, it was already stored and removed.
                    operations.append(['delete', p])
                    in_fast.remove(p)
                    
    return operations

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

def _pebble_greedy(dag, fast_memory_budget, deadline):
    """
    Greedy pebbling with reference counting and eviction.
    """
    start_time = time.time()
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    topo = _topological_order(dag)
    inputs = set(_get_inputs(dag))
    outputs = set(_get_outputs(dag))
    
    # Precompute children and reference counts
    children = defaultdict(list)
    ref_count = defaultdict(int)
    for v in all_v:
        ps = _get_parents(dag, v)
        for p in ps:
            children[p].append(v)
            ref_count[p] += 1
    
    operations = []
    in_fast = set()
    # next_use[v] = index in topo order of the next child that needs v
    next_use = {}
    
    # Precompute next_use for each vertex
    for v in all_v:
        ch = children[v]
        if ch:
            min_idx = min(topo.index(c) for c in ch)
            next_use[v] = min_idx
        else:
            next_use[v] = len(topo)
            
    # Simulation
    for i, v in enumerate(topo):
        if time.time() - deadline[0] > _DEADLINE:
            break
            
        parents = _get_parents(dag, v)
        
        if not parents:
            # Input vertex
            if children[v]:
                if v not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        candidates = []
                        for u in in_fast:
                            if u not in parents:
                                nu = next_use.get(u, len(topo))
                                candidates.append((nu, u))
                        
                        if candidates:
                            candidates.sort(key=lambda x: -x[0])
                            victim = candidates[0][1]
                            operations.append(['delete', victim])
                            in_fast.remove(victim)
                        
                    operations.append(['load', v])
                    in_fast.add(v)
            else:
                if v in outputs:
                    if v not in in_fast:
                        if len(in_fast) >= fast_memory_budget:
                             candidates = []
                             for u in in_fast:
                                nu = next_use.get(u, len(topo))
                                candidates.append((nu, u))
                             if candidates:
                                candidates.sort(key=lambda x: -x[0])
                                victim = candidates[0][1]
                                operations.append(['delete', victim])
                                in_fast.remove(victim)
                        operations.append(['load', v])
                        in_fast.add(v)
                    operations.append(['store', v])
                    in_fast.remove(v)
            
        else:
            # Internal vertex
            for p in parents:
                if p not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        candidates = []
                        for u in in_fast:
                            if u not in parents:
                                nu = next_use.get(u, len(topo))
                                candidates.append((nu, u))
                        
                        if candidates:
                            candidates.sort(key=lambda x: -x[0])
                            victim = candidates[0][1]
                            operations.append(['delete', victim])
                            in_fast.remove(victim)
                        
                    operations.append(['load', p])
                    in_fast.add(p)
            
            operations.append(['compute', v])
            in_fast.add(v)
            
            if v in outputs:
                operations.append(['store', v])
                in_fast.remove(v)
            else:
                if not children[v]:
                    operations.append(['delete', v])
                    in_fast.remove(v)
    
    return operations

def pebble(dag, fast_memory_budget):
    deadline = [time.time()]
    
    try:
        # Try layered strategy first
        ops_layered = _pebble_layered(dag, fast_memory_budget, deadline)
        if ops_layered is not None:
            if _validate_operations(ops_layered, dag, fast_memory_budget):
                if len(ops_layered) <= 20000:
                    return ops_layered
                    
        # Fallback to greedy
        ops_greedy = _pebble_greedy(dag, fast_memory_budget, deadline)
        
        if not ops_greedy:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        
        if _validate_operations(ops_greedy, dag, fast_memory_budget):
            if len(ops_greedy) <= 20000:
                return ops_greedy
        
        # If invalid, fallback
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
        
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
# EVOLVE-BLOCK-END
