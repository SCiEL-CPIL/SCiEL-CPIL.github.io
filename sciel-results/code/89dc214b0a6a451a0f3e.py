"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    dag: dict with keys:
      'vertices': list of vertex ids
      'edges': list of (parent, child) or similar
      'inputs': list of input vertex ids
      'outputs': list of output vertex ids
      'parents': dict vertex -> list of parent vertices (topologically ordered)
    
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    # Extract structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        edges = dag.get('edges', [])
    else:
        # Try to handle other formats
        vertices = list(dag.keys()) if hasattr(dag, 'keys') else []
        parents = {}
        inputs = set()
        outputs = set()
        edges = []
    
    if not vertices:
        return []
    
    # Build parent/children maps
    if not parents:
        parents = defaultdict(list)
        children = defaultdict(list)
        for u, v in edges:
            parents[v].append(u)
            children[u].append(v)
    else:
        parents = {k: list(v) for k, v in parents.items()}
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
    
    # Topological order
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    queue = [v for v in vertices if in_degree[v] == 0]
    topo_order = []
    temp_indeg = dict(in_degree)
    from collections import deque
    q = deque(queue)
    while q:
        u = q.popleft()
        topo_order.append(u)
        for c in children.get(u, []):
            temp_indeg[c] -= 1
            if temp_indeg[c] == 0:
                q.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle? Just use given order
        topo_order = list(vertices)
    
    # State tracking
    # 'blue' = on disk (available, no memory cost)
    # 'red' = in fast memory (costs 1 slot)
    # 'uncomputed' = not yet computed
    state = {v: 'uncomputed' for v in vertices}
    for v in inputs:
        state[v] = 'blue'  # inputs start blue
    
    # Track which vertices are in fast memory
    fast_mem = set()
    operations = []
    
    # Count how many times each vertex is used as a parent (fan-out)
    fan_out = {v: len(children.get(v, [])) for v in vertices}
    
    # Position in topological order for lookahead
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    # For each vertex, when is it last needed?
    last_needed = {}
    for v in vertices:
        if v in outputs:
            last_needed[v] = topo_pos.get(v, 0)
        elif fan_out[v] > 0:
            # Last child's position
            last_child_pos = max(topo_pos.get(c, 0) for c in children.get(v, []))
            last_needed[v] = last_child_pos
        else:
            last_needed[v] = topo_pos.get(v, 0)
    
    # Process in topological order
    for v in topo_order:
        if state[v] == 'blue' and v in outputs:
            # Already blue and is output, done
            continue
        
        if state[v] == 'red':
            # Already in fast memory
            if v in outputs:
                # Need to make it blue at the end, but for now keep it
                pass
            continue
        
        if state[v] == 'blue' and v not in inputs and v not in outputs:
            # Blue but not output - already computed and stored
            # We might need to reload it for computation of children? 
            # Actually if it's blue, it's already computed. But to compute children,
            # we need it red or we load it.
            # Actually, blue means it's on disk. To use it as a parent, we need to load it.
            # But we only need to load it when we compute a child.
            # So skip for now.
            continue
        
        # Need to compute v
        # Ensure all parents are available (red or blue -> load if blue)
        for p in parents.get(v, []):
            if state[p] == 'uncomputed':
                # Should not happen in topological order, but handle
                pass
            elif state[p] == 'blue':
                # Load parent into fast memory
                operations.append(['load', p])
                state[p] = 'red'
                fast_mem.add(p)
            elif state[p] == 'red':
                # Already in fast memory
                pass
        
        # Compute v
        operations.append(['compute', v])
        state[v] = 'red'
        fast_mem.add(v)
        
        # Now decide memory management
        # If fast memory is over budget, evict some values
        while len(fast_mem) > fast_memory_budget:
            # Find which value to evict
            # Heuristic: evict the one with the smallest last_needed (earliest last use)
            # But we need to keep v (just computed) if possible
            # Also, we should not evict values that are needed for the current computation
            # Actually, we just computed v, so its parents are still needed for other children
            
            # Score each value in fast_mem by last_needed
            # Evict the one with earliest last_needed that is not v
            candidates = [x for x in fast_mem if x != v]
            if not candidates:
                # Only v is in fast memory, but budget is 0?
                # Store v
                operations.append(['store', v])
                state[v] = 'blue'
                fast_mem.remove(v)
                break
            
            # Pick the one with earliest last_needed
            # But also consider: if a value is an output, we might want to keep it
            # Actually outputs just need to be blue at the end
            def eviction_score(x):
                # Lower score = evict first
                # Prefer to evict things that are used soonest
                score = last_needed.get(x, 0)
                # If it's an input, it can be reloaded cheaply
                if x in inputs:
                    score -= 1000  # Strongly prefer evicting inputs
                # If it has high fan-out, it's more expensive to recompute
                if fan_out[x] > 1:
                    score += 500 * fan_out[x]  # Keep high fan-out values
                return score
            
            # Sort by score, evict lowest
            candidates.sort(key=eviction_score)
            evict = candidates[0]
            
            if state[evict] == 'red':
                operations.append(['store', evict])
                state[evict] = 'blue'
                fast_mem.discard(evict)
        
        # If v is not an output and fast memory has room, keep it red
        # If v is an output, we'll store it at the end
    
    # Ensure all outputs are blue
    for v in outputs:
        if state[v] == 'red':
            operations.append(['store', v])
            state[v] = 'blue'
            fast_mem.discard(v)
    
    # Optional: delete any remaining red values to free memory (not required but clean)
    # The problem says "every red value consumes one fast-memory slot"
    # We should probably not leave extra red values
    for v in list(fast_mem):
        if state[v] == 'red' and v not in outputs:
            operations.append(['delete', v])
            state[v] = 'uncomputed'
            fast_mem.discard(v)
    
    return operations
# EVOLVE-BLOCK-END

