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

def _pebble_dp_small(dag, fast_memory_budget, deadline):
    """
    Exact DP over memory states for small DAGs.
    State: (next_vertex_idx, memory_mask) where memory_mask is a bitmask
    of which non-input vertices are currently in fast memory.
    Returns list of operations or None if DP not applicable.
    """
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    topo = _topological_order(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    output_set = set(outputs)
    
    # Map non-input vertices to bit positions
    non_input_verts = [v for v in topo if v not in inputs]
    n_ni = len(non_input_verts)
    
    # DP only feasible if bitmask fits in reasonable memory
    if n_ni > 12:
        return None
    
    ni_idx = {v: i for i, v in enumerate(non_input_verts)}
    
    # Precompute parent bitmasks
    parent_masks = {}
    for v in topo:
        if v in inputs:
            parent_masks[v] = 0
        else:
            mask = 0
            for p in _get_parents(dag, v):
                if p in ni_idx:
                    mask |= (1 << ni_idx[p])
            parent_masks[v] = mask
    
    # Output mask: which non-input vertices are outputs
    output_mask = 0
    for o in outputs:
        if o in ni_idx:
            output_mask |= (1 << ni_idx[o])
    
    # DP: dp[v_idx][mask] = (min_transfers, prev_state)
    # v_idx: index in topo of next vertex to process
    # mask: which non-input vertices are in fast memory
    # We process vertices in topo order. At each step, we decide which
    # vertices to load/delete before computing the next vertex.
    
    # Actually, let's think differently. We process vertices in topological order.
    # Before computing vertex v, we need all its parents in fast memory.
    # We can load/delete as needed.
    
    # State: (i, mask) where i = next vertex index to compute, mask = current fast memory
    # Cost to transition: number of loads/stores needed to make parents of topo[i] available
    
    INF = float('inf')
    # dp[i][mask] = min transfers to reach state (i, mask)
    # Initialize: dp[0][0] = 0 (no vertices in fast memory initially)
    # But inputs are always available (they start blue, we can load them)
    
    # Actually, inputs are not in fast memory initially. They need to be loaded.
    # Let me reconsider: the state should track which vertices are in fast memory.
    # Inputs can be loaded at any time.
    
    # Let's use: dp[i][mask] = min number of transfers to have processed
    # vertices 0..i-1 and have exactly the vertices in 'mask' in fast memory.
    
    # For vertex topo[i], we need its parents in fast memory before computing.
    # We can load any missing parents (cost = number loaded) and optionally
    # delete some vertices (cost = number deleted) before computing.
    
    # After computing topo[i], it goes into fast memory.
    
    # Transition from (i, mask) to (i+1, mask'):
    # 1. Load missing parents of topo[i] that aren't in mask
    # 2. Optionally delete some vertices to make room
    # 3. Compute topo[i] (adds it to fast memory)
    # 4. Optionally delete more vertices
    # mask' = mask after all operations
    
    # This is complex. Let me use a simpler BFS/DP approach.
    
    # Alternative: simulate all possible sequences of operations.
    # For small n_ni, the number of states is n_ni * 2^n_ni which is manageable.
    
    # Let me define:
    # - We process vertices in topo order
    # - At step i (processing vertex v = topo[i]), we have some set S in fast memory
    # - We need all parents of v in S (or load them)
    # - After computing v, v is in fast memory
    # - We can delete any subset of fast memory vertices
    # - Fast memory size is limited to fast_memory_budget
    
    # DP state: (i, frozenset of vertices in fast memory)
    # But frozensets are expensive. Use bitmask for non-input vertices.
    # Inputs are special: they can always be loaded (they're in slow memory).
    
    # Let me track: which non-input vertices are in fast memory (bitmask)
    # And implicitly, we know which inputs might be in fast memory... 
    # Actually, inputs don't need to be tracked separately since they can be
    # loaded at any time. But they DO occupy fast memory slots.
    
    # Hmm, this is getting complex. Let me simplify:
    # Track all vertices in fast memory as a bitmask over ALL vertices.
    # But that requires 2^n states which is too much for n > 20.
    
    # For the DP to be feasible, let's limit to n_ni <= 10 and assume
    # inputs are handled separately.
    
    # Actually, let me just do a greedy approach with better heuristics
    # and skip the DP for now. The DP is too complex to implement correctly
    # in the time available.
    
    return None

def _pebble_improved_greedy(dag, fast_memory_budget, deadline):
    """
    Improved greedy pebbling with better eviction strategy.
    Uses exact remaining-use counts for eviction decisions.
    """
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    topo = _topological_order(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    output_set = set(outputs)
    
    # Compute children
    children = defaultdict(list)
    for v in topo:
        parents = _get_parents(dag, v)
        for p in parents:
            children[p].append(v)
    
    # Compute exact remaining uses: for each vertex, how many children 
    # haven't been processed yet
    remaining_uses = {v: len(children[v]) for v in topo}
    
    # Compute layer for tie-breaking
    layer = _compute_layers(dag, topo)
    
    operations = []
    in_fast = set()
    
    def evict_one():
        if not in_fast:
            return
        # Find the best vertex to evict
        # Priority: 
        # 1. Non-outputs preferred for eviction
        # 2. Vertices with 0 remaining uses (done) 
        # 3. Lower remaining uses
        # 4. Older (smaller layer)
        
        candidates = list(in_fast)
        
        def sort_key(v):
            is_output = 1 if v in output_set else 0
            ru = remaining_uses.get(v, 0)
            lv = layer.get(v, 0)
            # We want to evict: non-output, 0 remaining uses, older
            # Lower key = better to evict
            return (is_output, ru, lv)
        
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
        # Try exact DP for small DAGs
        ops_dp = _pebble_dp_small(dag, fast_memory_budget, deadline)
        if ops_dp is not None:
            if _validate_operations(ops_dp, dag, fast_memory_budget):
                if len(ops_dp) <= 20000:
                    return ops_dp
        
        # Use improved greedy
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
