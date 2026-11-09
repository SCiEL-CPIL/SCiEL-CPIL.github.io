"""DAG pebbling with communication minimization."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize communication (transfers).
    
    Args:
        dag: dict with keys:
            - 'vertices': list of vertex IDs
            - 'edges': list of (parent, child) tuples
            - 'inputs': list of input vertex IDs (start blue)
            - 'outputs': list of output vertex IDs (must finish blue)
        fast_memory_budget: int, number of fast memory slots available
    
    Returns:
        list of [operation, vertex] where operation is one of:
            'load', 'store', 'compute', 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse DAG structure
    vertices = list(dag.get('vertices', []))
    edges = list(dag.get('edges', []))
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build adjacency
    parents = defaultdict(list)
    children = defaultdict(list)
    in_degree = defaultdict(int)
    
    for u, v in edges:
        children[u].append(v)
        parents[v].append(u)
        in_degree[v] += 1
    
    # Topological sort
    topo_order = []
    queue = deque()
    for v in vertices:
        if in_degree[v] == 0:
            queue.append(v)
    
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for w in children[v]:
            in_degree[w] -= 1
            if in_degree[w] == 0:
                queue.append(w)
    
    # If topological sort didn't cover all vertices, add remaining
    if len(topo_order) < n:
        visited = set(topo_order)
        for v in vertices:
            if v not in visited:
                topo_order.append(v)
    
    # Identify input and output sets
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Determine which vertices are inputs (no parents) and outputs (no children or explicitly marked)
    for v in vertices:
        if not parents[v]:
            input_set.add(v)
        if not children[v]:
            output_set.add(v)
    
    # Simple greedy strategy: process in topological order,
    # load inputs, compute when all parents are available,
    # store outputs, delete when no longer needed.
    
    # Track state: 'blue' (in fast memory), 'red' (in slow memory), 'gone'
    state = {}  # vertex -> 'blue', 'red', 'gone'
    for v in vertices:
        if v in input_set:
            state[v] = 'blue'  # inputs start blue
        else:
            state[v] = 'gone'
    
    fast_mem = set()  # vertices currently in fast memory (blue)
    ops = []
    
    # Count how many children need each vertex (to know when we can delete)
    remaining_children = defaultdict(int)
    for v in vertices:
        for w in children[v]:
            remaining_children[v] += 1
    
    # For each vertex, track how many parents still need to be computed/available
    uncomputed_parents = {}
    for v in vertices:
        uncomputed_parents[v] = len(parents[v])
    
    # Process in topological order
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in input_set:
            # Inputs are already blue, no action needed
            # But if they're not in fast memory yet, they should be
            if v not in fast_mem:
                # This shouldn't happen for inputs, but handle gracefully
                pass
            continue
        
        # Check if all parents are available (blue or red)
        all_parents_ready = True
        for p in parents[v]:
            if state.get(p, 'gone') == 'gone':
                all_parents_ready = False
                break
        
        if not all_parents_ready:
            continue
        
        # Need to compute v
        # Check if we have fast memory space
        if len(fast_mem) >= fast_memory_budget:
            # Need to free up space: delete a vertex that's no longer needed
            freed = False
            for u in list(fast_mem):
                if u == v:
                    continue
                # Can we delete u?
                if remaining_children[u] == 0 and u not in output_set:
                    # Safe to delete
                    ops.append(['delete', u])
                    fast_mem.discard(u)
                    state[u] = 'gone'
                    freed = True
                    break
                # Also check if u is an output that's already been stored
                if u in output_set and state.get(u) == 'red':
                    # Already stored, can delete from fast mem
                    ops.append(['delete', u])
                    fast_mem.discard(u)
                    state[u] = 'red'
                    freed = True
                    break
            
            if not freed:
                # Force delete the oldest non-output, non-needed vertex
                for u in list(fast_mem):
                    if u not in output_set and remaining_children[u] == 0:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                        state[u] = 'gone'
                        freed = True
                        break
            
            if not freed and fast_memory_budget > 0:
                # Last resort: delete any non-output vertex
                for u in list(fast_mem):
                    if u not in output_set:
                        ops.append(['delete', u])
                        fast_mem.discard(u)
                        state[u] = 'gone'
                        freed = True
                        break
        
        # Now compute v
        ops.append(['compute', v])
        fast_mem.add(v)
        state[v] = 'blue'
        
        # Update uncomputed_parents for children
        for w in children[v]:
            uncomputed_parents[w] -= 1
        
        # If v is an output, store it
        if v in output_set:
            ops.append(['store', v])
            state[v] = 'red'
            # We can keep it in fast mem or not; for now keep it
            # Actually, after storing, we can delete from fast mem if needed
            # But let's keep it for now and delete later if space is needed
        
        # Decrement remaining_children for v's parents? No, remaining_children
        # tracks how many children still need v. When v is computed, its children
        # can potentially be computed.
    
    # Ensure all outputs are blue at the end
    for v in output_set:
        if time.time() - start_time > DEADLINE:
            break
        if state.get(v, 'gone') != 'blue':
            if state.get(v) == 'red':
                ops.append(['load', v])
                fast_mem.add(v)
                state[v] = 'blue'
            else:
                # Need to recompute - this is complex, skip for now
                pass
    
    return ops

# EVOLVE-BLOCK-END
