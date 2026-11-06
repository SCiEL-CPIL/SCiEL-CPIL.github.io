def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    def elapsed():
        return time.time() - start_time

    # Extract DAG structure robustly
    layers = None
    parents_raw = None
    inputs = None
    outputs = None
    vertices = None

    if isinstance(dag, dict):
        layers = dag.get('layers', None)
        parents_raw = dag.get('parents', None)
        inputs = dag.get('inputs', None)
        outputs = dag.get('outputs', None)
        vertices = dag.get('vertices', None)
    elif isinstance(dag, list):
        # Assume dag is list of layers
        layers = dag
        # parents, inputs, outputs might be in a separate config or inferred
        # But usually if dag is a list, it's just the layers.
        # We need to find parents. If not provided, we might have to infer or assume a specific structure.
        # However, the contract says "dag has topologically ordered parents and output vertices".
        # This implies the dag object *contains* this info.
        # If dag is a list, maybe it's a list of vertex definitions?
        # Let's check common formats.
        # Format 1: List of layers. Each layer is a list of vertex IDs.
        # Format 2: List of dicts?
        # Given the error, let's assume the previous code failed because it treated a list as a dict.
        
        # If dag is a list, it's likely layers.
        # But we need parents.
        # If parents is not in the list, we can't compute.
        # Perhaps the task provides a dict, and the error was due to a specific key being a list?
        
        # Let's look at the error again: 'list' object has no attribute 'keys'.
        # This usually happens when you do `some_list.keys()`.
        # In the previous code, `parents` was accessed with `.get` or iterated.
        # If `parents` was a list, `parents.get` fails with 'list' object has no attribute 'get'.
        # The error is 'keys'.
        # Maybe `dag` itself was iterated?
        # `for v, ps in dag.items():`? No, the code doesn't do that.
        
        # Wait, if `dag` is a dict, and `dag['parents']` is a list, and the code does `parents.keys()`?
        # The provided code does NOT do `parents.keys()`.
        # It does `parents.get(v, [])`.
        
        # Is it possible the error is from `vertices`?
        # `vertices = dag.get('vertices', None)`
        
        # Let's assume the provided code is just a snippet and the actual execution hit a line not shown or I am misinterpreting the standard library behavior.
        # Actually, if `parents` is a list, and I do `for v in parents:`, that works.
        # If I do `parents[v]`, that works if v is int.
        
        # Let's just make the code robust to `parents` being a list or dict.
        pass
    else:
        # Try attributes
        layers = getattr(dag, 'layers', None)
        parents_raw = getattr(dag, 'parents', None)
        inputs = getattr(dag, 'inputs', None)
        outputs = getattr(dag, 'outputs', None)
        vertices = getattr(dag, 'vertices', None)

    # Normalize parents to a dict: {vertex: [parent1, parent2, ...]}
    parents = {}
    
    if parents_raw is not None:
        if isinstance(parents_raw, dict):
            parents = {v: list(ps) for v, ps in parents_raw.items()}
        elif isinstance(parents_raw, list):
            # Assume parents_raw[i] is the list of parents for vertex i
            # We need to know the vertex IDs.
            # If vertices is a list, vertices[i] is the ID.
            # If vertices is None, assume ID is i.
            if vertices is not None and isinstance(vertices, list):
                for i, vid in enumerate(vertices):
                    if i < len(parents_raw):
                        parents[vid] = list(parents_raw[i])
            else:
                for i in range(len(parents_raw)):
                    parents[i] = list(parents_raw[i])
        else:
            # Unknown type, try to treat as dict-like?
            try:
                parents = {v: list(ps) for v, ps in parents_raw.items()}
            except:
                parents = {}
    else:
        # No parents provided, must infer from layers or return empty
        if layers is None:
            return []
        # Infer parents from layers? 
        # Usually, if layers are given, edges go from layer i to i+1.
        # But we don't know which vertices connect to which without edge info.
        # If no parents and no edges, we can't compute.
        # However, if it's a "layered DAG" as per contract, maybe we assume full connectivity between adjacent layers?
        # "Score mean transfers / naive schedule on three layered DAGs"
        # "dag has topologically ordered parents"
        # This implies parents ARE provided.
        
        # If parents is None, we might be in a case where dag is a list of layers and parents is implicit?
        # Or maybe the key is 'edges'?
        # Let's check if 'edges' was in the dict.
        if isinstance(dag, dict) and 'edges' in dag:
            edges = dag['edges']
            # Edges might be list of (u, v)
            for u, v in edges:
                if v not in parents:
                    parents[v] = []
                parents[v].append(u)
        
        if not parents:
            # If we still don't have parents, we can't do much.
            # But we need to handle the case where the program crashed before.
            # If we can't determine structure, return empty.
            return []

    # Ensure all vertices in layers are in parents dict (with empty list if leaf/input)
    if layers is not None:
        all_vertices_in_layers = set()
        for layer in layers:
            for v in layer:
                all_vertices_in_layers.add(v)
        for v in all_vertices_in_layers:
            if v not in parents:
                parents[v] = []
    else:
        # Compute layers from parents
        # Topological sort
        in_degree = {v: len(ps) for v, ps in parents.items()}
        children = {v: [] for v in parents}
        for v, ps in parents.items():
            for p in ps:
                if p not in children:
                    children[p] = []
                children[p].append(v)
        
        from collections import deque
        queue = deque([v for v, deg in in_degree.items() if deg == 0])
        layers = []
        while queue:
            layer = []
            for _ in range(len(queue)):
                v = queue.popleft()
                layer.append(v)
                for c in children.get(v, []):
                    in_degree[c] -= 1
                    if in_degree[c] == 0:
                        queue.append(c)
            if layer:
                layers.append(layer)
        
        if inputs is None:
            inputs = layers[0] if layers else []
        if outputs is None:
            outputs = layers[-1] if layers else []
        if vertices is None:
            vertices = [v for layer in layers for v in layer]

    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = []
        
    # Convert inputs/outputs to sets for faster lookup
    inputs_set = set(inputs)
    outputs_set = set(outputs)

    # Strategy:
    # The task asks for communication-minimizing DAG pebbling.
    # Naive schedule: Load all inputs, compute layer by layer, store results when not needed.
    # We need to minimize load/store operations.
    
    # For a layered DAG, a good heuristic is:
    # 1. Load inputs.
    # 2. For each layer L_i:
    #    a. Ensure parents of L_i are in memory.
    #    b. Compute vertices in L_i.
    #    c. Store vertices from previous layers that are no longer needed.
    #    d. If memory is full, store vertices from current layer that are not needed for future layers (or recompute later if necessary).
    
    # "Recomputations allowed"
    # If we run out of memory, we can store a value and recompute it later if needed.
    # However, recomputation costs compute operations, which are not penalized as heavily as transfers?
    # "Score mean transfers / naive schedule"
    # Transfers are load/store. Compute is free? Or just not counted in the score?
    # "every red value consumes one fast-memory slot" -> Red = not in memory?
    # "Inputs start blue, all outputs must finish blue" -> Blue = in memory.
    # So we want to minimize Load/Store operations.
    
    # Let's implement a greedy layer-by-layer strategy with memory management.
    
    operations = []
    in_memory = set(inputs)
    
    # Precompute: for each vertex, the last layer index where it is needed as a parent.
    vertex_last_needed = {}
    for i, layer in enumerate(layers):
        for v in layer:
            for p in parents.get(v, []):
                if p not in vertex_last_needed or i > vertex_last_needed[p]:
                    vertex_last_needed[p] = i
    
    # Also, outputs must be in memory at the end.
    # So effectively, outputs are "needed" until the end.
    # Let's treat outputs as having last_needed = len(layers)
    for v in outputs:
        if v not in vertex_last_needed or len(layers) > vertex_last_needed[v]:
            vertex_last_needed[v] = len(layers)

    # Process layers
    for i, layer in enumerate(layers):
        if elapsed() > DEADLINE:
            break
            
        # 1. Ensure parents of current layer are in memory
        for v in layer:
            for p in parents.get(v, []):
                if p not in in_memory:
                    # Check if we can free something to make space?
                    # Or just load it?
                    # If we load it, we might exceed budget.
                    # We should store something first if over budget.
                    if len(in_memory) >= fast_memory_budget:
                        # Find a candidate to store
                        # Prefer vertices not needed in future layers
                        # Prefer vertices not in current layer (if possible)
                        # Prefer vertices with smallest last_needed
                        candidates = [x for x in in_memory if x not in outputs_set]
                        if candidates:
                            # Sort by last_needed
                            candidates.sort(key=lambda x: vertex_last_needed.get(x, -1))
                            # Store the one with earliest last_needed
                            to_store = candidates[0]
                            operations.append(['store', to_store])
                            in_memory.discard(to_store)
                    operations.append(['load', p])
                    in_memory.add(p)
        
        # 2. Compute vertices in current layer
        for v in layer:
            if v not in in_memory:
                # All parents should be in memory now
                operations.append(['compute', v])
                in_memory.add(v)
                
        # 3. Clean up: Store vertices that are no longer needed
        # A vertex can be stored if:
        # - It is not an output
        # - Its last_needed < i+1 (i.e., not needed in current or future layers? 
        #   Wait, if it was a parent of current layer, it was needed in layer i.
        #   If vertex_last_needed[v] == i, it is NOT needed in layer i+1.
        #   So after computing layer i, we can store v if vertex_last_needed[v] <= i)
        #   Actually, if vertex_last_needed[v] == i, it was used in layer i. 
        #   It is not used in layer i+1. So we can store it now.
        
        to_store = []
        for v in list(in_memory):
            if v in outputs_set:
                continue
            # If v is in the current layer, we just computed it.
            # We can store it if it's not needed later.
            # If v is from a previous layer, we can store it if it's not needed later.
            
            last_need = vertex_last_needed.get(v, -1)
            # If last_need < i, it wasn't even needed in current layer (should have been stored earlier)
            # If last_need == i, it was needed in current layer, but not in i+1.
            # So if last_need <= i, we can store it.
            if last_need <= i:
                to_store.append(v)
                
        # If we are still over budget, store more
        if len(in_memory) > fast_memory_budget:
            # Find more candidates
            # Sort remaining in_memory (not in to_store, not outputs) by last_needed
            remaining = [v for v in in_memory if v not in to_store and v not in outputs_set]
            remaining.sort(key=lambda x: vertex_last_needed.get(x, -1))
            for v in remaining:
                if len(in_memory) <= fast_memory_budget:
                    break
                to_store.append(v)
                
        for v in to_store:
            if v in in_memory:
                operations.append(['store', v])
                in_memory.discard(v)
                
    # Ensure outputs are in memory
    for v in outputs:
        if v not in in_memory:
            # Recompute or Load
            # If parents are in memory, compute
            # Else, load parents?
            # For simplicity, if we failed to keep it, we might need to recompute.
            # But recomputation requires parents.
            # This is a fallback.
            # If v is an input, load it.
            if v in inputs_set:
                operations.append(['load', v])
                in_memory.add(v)
            else:
                # Try to compute
                # Ensure parents are in memory
                for p in parents.get(v, []):
                    if p not in in_memory:
                        # Load p
                        if len(in_memory) >= fast_memory_budget:
                            # Store something
                            candidates = [x for x in in_memory if x not in outputs_set and x != v]
                            if candidates:
                                candidates.sort(key=lambda x: vertex_last_needed.get(x, -1))
                                to_store = candidates[0]
                                operations.append(['store', to_store])
                                in_memory.discard(to_store)
                        operations.append(['load', p])
                        in_memory.add(p)
                operations.append(['compute', v])
                in_memory.add(v)
                
    return operations
