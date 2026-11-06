"""Rewritten program for communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize load/store transfers.
    
    dag: dict with keys 'vertices', 'parents', 'outputs'
         - vertices: list of vertex IDs (topologically ordered)
         - parents: dict mapping vertex -> list of parent vertex IDs
         - outputs: list of output vertex IDs
    fast_memory_budget: int, number of fast memory slots
    
    Returns: list of [operation, vertex] where operation is one of
             'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    vertices = dag['vertices']
    parents = dag['parents']
    outputs = set(dag['outputs'])
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build adjacency and compute in-degree (for topological order)
    # parents already gives us the structure
    # children: vertex -> list of children
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Compute number of uses for each vertex (how many children use it)
    # This helps decide which values to keep in fast memory
    use_count = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            use_count[p] += 1
    
    # Also count if vertex is an output (must end blue)
    for v in outputs:
        use_count[v] += 1  # outputs must be kept until end
    
    # Topological order (vertices are already topologically ordered per task)
    topo_order = list(vertices)
    
    # Track state: 'blue' (in fast memory), 'red' (on disk), or not computed
    # Initially, inputs (vertices with no parents) are blue
    state = {}  # vertex -> 'blue' or 'red' or None (not computed)
    for v in vertices:
        if len(parents.get(v, [])) == 0:
            state[v] = 'blue'
        else:
            state[v] = None
    
    # Fast memory set
    fast_mem = set(v for v in vertices if state[v] == 'blue')
    
    # Operations list
    ops = []
    
    # Process in topological order
    for v in topo_order:
        if time.time() > DEADLINE:
            # Return what we have (may be incomplete but valid so far)
            break
        
        # Check if all parents are available (blue)
        parents_of_v = parents.get(v, [])
        all_parents_blue = all(state.get(p) == 'blue' for p in parents_of_v)
        
        if not all_parents_blue:
            # Need to load some parents
            for p in parents_of_v:
                if state.get(p) == 'red':
                    # Load parent into fast memory
                    if len(fast_mem) >= fast_memory_budget:
                        # Need to evict something
                        # Evict a value that won't be needed soon or has low use count
                        evicted = _evict(fast_mem, use_count, topo_order, v, state, outputs)
                        if evicted is not None:
                            ops.append(['delete', evicted])
                            state[evicted] = 'red'
                            fast_mem.discard(evicted)
                        else:
                            # Should not happen if budget >= 1
                            break
                    ops.append(['load', p])
                    state[p] = 'blue'
                    fast_mem.add(p)
                elif state.get(p) is None:
                    # Parent not computed yet - shouldn't happen in topo order
                    pass
        
        # Now compute v
        if state.get(v) is None:
            # Check if fast memory has space
            if len(fast_mem) >= fast_memory_budget:
                # Evict something
                evicted = _evict(fast_mem, use_count, topo_order, v, state, outputs)
                if evicted is not None:
                    ops.append(['delete', evicted])
                    state[evicted] = 'red'
                    fast_mem.discard(evicted)
            
            ops.append(['compute', v])
            state[v] = 'blue'
            fast_mem.add(v)
        
        # After computing, consider storing values that are no longer needed
        # A value is "done" if all its children have been processed
        for p in parents_of_v:
            if state.get(p) == 'blue':
                # Check if all children of p have been processed
                all_children_done = all(
                    state.get(c) is not None for c in children[p]
                )
                if all_children_done and p not in outputs:
                    # Can store p
                    ops.append(['store', p])
                    state[p] = 'red'
                    fast_mem.discard(p)
    
    # At the end, ensure all outputs are blue
    for v in outputs:
        if state.get(v) == 'red':
            if len(fast_mem) >= fast_memory_budget:
                evicted = _evict(fast_mem, use_count, topo_order, -1, state, outputs)
                if evicted is not None:
                    ops.append(['delete', evicted])
                    state[evicted] = 'red'
                    fast_mem.discard(evicted)
            ops.append(['load', v])
            state[v] = 'blue'
            fast_mem.add(v)
    
    # Truncate if too many operations
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops


def _evict(fast_mem, use_count, topo_order, current_v, state, outputs):
    """
    Choose a vertex to evict from fast memory.
    Strategy: evict the vertex with lowest future use count,
    preferring non-output vertices.
    """
    if not fast_mem:
        return None
    
    # Find the best candidate to evict
    # Score: lower is better to evict
    # Prefer evicting vertices that are not outputs and have low use count
    best = None
    best_score = float('inf')
    
    for v in fast_mem:
        if v in outputs:
            # Don't evict outputs if possible
            continue
        # Use count as proxy for future need
        score = use_count.get(v, 0)
        if score < best_score:
            best_score = score
            best = v
    
    if best is None:
        # All are outputs, evict the one with lowest use count
        for v in fast_mem:
            score = use_count.get(v, 0)
            if score < best_score:
                best_score = score
                best = v
    
    return best
# EVOLVE-BLOCK-END
