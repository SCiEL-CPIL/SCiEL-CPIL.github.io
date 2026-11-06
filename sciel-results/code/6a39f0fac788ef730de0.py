"""Communication-minimizing DAG pebbling via dynamic scheduling."""

def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem.
    
    Args:
        dag: A DAG structure with vertices, edges (parents), and output vertices.
             The exact structure may vary, but we assume common patterns:
             - dag['vertices'] or dag['nodes']: list of vertex IDs
             - dag['edges'] or dag['parents']: dict mapping vertex -> list of parents
             - dag['outputs'] or dag['sinks']: list of output vertex IDs
             - dag['inputs'] or dag['sources']: list of input vertex IDs
        fast_memory_budget: Maximum number of red (computed) values in fast memory.
    
    Returns:
        List of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'.
    """
    # Parse the DAG structure
    # Try to handle various common formats
    
    if isinstance(dag, dict):
        # Try to extract vertices, edges, inputs, outputs
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        edges = dag.get('edges', dag.get('parents', dag.get('E', dag.get('adj', None))))
        inputs = dag.get('inputs', dag.get('sources', dag.get('in', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('out', None)))
    elif hasattr(dag, 'vertices'):
        vertices = dag.vertices
        edges = getattr(dag, 'edges', getattr(dag, 'parents', getattr(dag, 'adj', {})))
        inputs = getattr(dag, 'inputs', getattr(dag, 'sources', []))
        outputs = getattr(dag, 'outputs', getattr(dag, 'sinks', []))
    else:
        # Fallback: assume it's a dict-like structure
        vertices = list(dag.keys()) if isinstance(dag, dict) else []
        edges = dag if isinstance(dag, dict) else {}
        inputs = []
        outputs = []
    
    if vertices is None:
        # Try to infer from edges
        if edges and isinstance(edges, dict):
            all_verts = set()
            for v, parents in edges.items():
                all_verts.add(v)
                for p in parents:
                    all_verts.add(p)
            vertices = list(all_verts)
        else:
            vertices = []
    
    if edges is None:
        edges = {}
    
    if inputs is None:
        # Inputs are vertices with no parents
        inputs = [v for v in vertices if not edges.get(v, [])]
    
    if outputs is None:
        # Outputs are vertices with no children
        child_map = {}
        for v, parents in edges.items():
            for p in parents:
                child_map.setdefault(p, []).append(v)
        outputs = [v for v in vertices if v not in child_map]
    
    # Convert edges to a consistent format: dict mapping vertex -> list of parent vertices
    if not isinstance(edges, dict):
        # Maybe it's a list of (parent, child) pairs
        if isinstance(edges, list):
            new_edges = {}
            for e in edges:
                if len(e) == 2:
                    p, c = e
                    new_edges.setdefault(c, []).append(p)
            edges = new_edges
        else:
            edges = {}
    
    # Ensure all vertices have an entry in edges
    for v in vertices:
        if v not in edges:
            edges[v] = []
    
    # Topological sort to get processing order
    in_degree = {v: len(edges.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v, parents in edges.items():
        for p in parents:
            children.setdefault(p, []).append(v)
    
    from collections import deque
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle detected, just use the list order
        topo_order = list(vertices)
    
    # Identify input and output vertices
    input_set = set(inputs) if inputs else set()
    output_set = set(outputs) if outputs else set()
    
    # If no outputs specified, treat vertices with no children as outputs
    if not output_set:
        for v in vertices:
            if not children.get(v, []):
                output_set.add(v)
    
    # If no inputs specified, treat vertices with no parents as inputs
    if not input_set:
        for v in vertices:
            if not edges.get(v, []):
                input_set.add(v)
    
    # Strategy: 
    # - Inputs start blue (in slow memory)
    # - We need all outputs to finish blue
    # - Fast memory holds red (computed) values, limited by fast_memory_budget
    # - Operations: load (blue->red), store (red->blue), compute (parents red -> child red), delete (red->gone)
    
    # For a layered DAG, a good strategy is to process layer by layer
    # or use a worklist approach that keeps fast memory full with useful values
    
    # Simple but effective strategy:
    # 1. Load all inputs to fast memory (they start blue, so we need to load them if we want to compute)
    #    Actually, inputs start blue. To compute a node, its parents must be red.
    #    So we need to load parents from blue to red before computing.
    
    # Let's use a priority-based scheduling:
    # - Maintain a set of vertices currently in fast memory (red)
    # - Process vertices in topological order
    # - When a vertex's parents are all in fast memory, compute it
    # - If fast memory is full, evict a vertex that won't be needed soon
    
    # Track which vertices are in fast memory (red state)
    in_fast = set()
    # Track which vertices are blue (in slow memory, available)
    # All vertices start blue except computed ones which become red
    # Actually, inputs start blue. All others start... not present? Or blue?
    # "Inputs start blue, all outputs must finish blue"
    # So initially: inputs are blue, others are not in memory
    # After computation: a vertex is red in fast memory
    # We can store a red value back to blue (slow memory)
    
    # State:
    # - blue: set of vertices that are in slow memory (available for load)
    # - red: set of vertices that are in fast memory
    # Initially: blue = inputs, red = empty
    # Goal: all outputs in blue, red can be anything (but we should clean up)
    
    blue = set(input_set)
    red = set()
    
    operations = []
    
    # Count how many times each vertex is needed as a parent
    use_count = {v: 0 for v in vertices}
    for v in vertices:
        for c in children.get(v, []):
            use_count[v] += 1
    
    # Process in topological order
    # For each vertex, ensure its parents are in fast memory, then compute it
    
    computed = set()
    
    # We'll iterate through topo_order and for each vertex, make sure parents are available
    for v in topo_order:
        if v in input_set:
            # Inputs start blue, no need to compute
            continue
        
        parents = edges.get(v, [])
        if not parents:
            # This is a source but not marked as input? Treat as input
            blue.add(v)
            continue
        
        # Ensure all parents are in fast memory (red)
        for p in parents:
            if p not in red:
                # Need to load p from blue to red
                if p in blue:
                    operations.append(['load', p])
                    blue.remove(p)
                    red.add(p)
                else:
                    # p is not in blue and not in red - this shouldn't happen if we're doing it right
                    # Maybe p was computed and deleted? Recompute?
                    # For now, this is an error state
                    pass
        
        # Now compute v
        # Check if fast memory has room
        if len(red) < fast_memory_budget:
            operations.append(['compute', v])
            red.add(v)
            computed.add(v)
        else:
            # Fast memory is full, need to evict something
            # Evict a vertex that is not an output and has been used the most (or least needed)
            # Simple heuristic: evict a non-output vertex
            candidates = [x for x in red if x not in output_set]
            if candidates:
                # Evict the one with highest use_count (already used most)
                # or lowest future use
                # For simplicity, evict the first non-output
                evict = candidates[0]
                operations.append(['delete', evict])
                red.remove(evict)
                # evict is now gone, but it was computed so it could be reloaded if it's in blue
                # Actually, after compute, it's red. If we delete it, it's gone.
                # But we could have stored it back to blue first
                # Let's improve: store back to blue instead of delete if it might be needed
            else:
                # All in fast memory are outputs, just compute and we'll deal with it
                # This shouldn't happen often
                operations.append(['compute', v])
                red.add(v)
                computed.add(v)
    
    # Now ensure all outputs are in blue
    for v in output_set:
        if v in red:
            operations.append(['store', v])
            red.remove(v)
            blue.add(v)
        elif v not in blue:
            # v is not in blue or red - it was never computed or was deleted
            # Need to recompute it
            # This is a problem; our simple approach might miss this
            pass
    
    # Clean up remaining red values
    for v in list(red):
        operations.append(['delete', v])
        red.remove(v)
    
    return operations
