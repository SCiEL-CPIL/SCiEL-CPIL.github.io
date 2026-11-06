def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    # --- Parsing ---
    # Support 3 formats:
    # 1. dag is a list of lists: dag[i] = parents of i
    # 2. dag is a dict with 'parents', 'inputs', 'outputs'
    # 3. dag is an object with .parents, .inputs, .outputs

    if isinstance(dag, list):
        n = len(dag)
        vertices = list(range(n))
        parents = {i: list(dag[i]) for i in range(n)}
        # Inputs are those with no parents
        inputs = set(i for i in vertices if len(parents[i]) == 0)
        # Build children
        children = {i: [] for i in vertices}
        for i in vertices:
            for p in parents[i]:
                children[p].append(i)
        # Outputs are those with no children
        outputs = set(i for i in vertices if len(children[i]) == 0)
        
    elif isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = list(dag['vertices'])
        else:
            # Infer from parents
            parents_dict = dag.get('parents', {})
            vertices = set()
            for v, ps in parents_dict.items():
                vertices.add(v)
                for p in ps:
                    vertices.add(p)
            vertices = list(vertices)
            # Ensure topological order? Dict keys might not be sorted.
            # We will sort topologically later.
            
        if 'parents' in dag:
            parents = {v: list(dag['parents'][v]) for v in vertices if v in dag['parents']}
            # Fill missing
            for v in vertices:
                if v not in parents:
                    parents[v] = []
        else:
            parents = {v: [] for v in vertices}
            
        if 'inputs' in dag:
            inputs = set(dag['inputs'])
        else:
            inputs = set(v for v in vertices if len(parents[v]) == 0)
            
        if 'outputs' in dag:
            outputs = set(dag['outputs'])
        else:
            children = {v: [] for v in vertices}
            for v, ps in parents.items():
                for p in ps:
                    children[p].append(v)
            outputs = set(v for v in vertices if len(children[v]) == 0)
            
        # Build children if not done
        if 'children' not in locals() or 'children' is None:
             children = {v: [] for v in vertices}
             for v, ps in parents.items():
                for p in ps:
                    children[p].append(v)

    else:
        # Assume object
        vertices = list(dag.vertices) if hasattr(dag, 'vertices') else []
        if not vertices and hasattr(dag, 'parents'):
            parents_dict = dag.parents
            vertices = set()
            for v, ps in parents_dict.items():
                vertices.add(v)
                for p in ps:
                    vertices.add(p)
            vertices = list(vertices)
            
        parents = {v: list(dag.parents[v]) for v in vertices if hasattr(dag, 'parents') and v in dag.parents}
        for v in vertices:
            if v not in parents:
                parents[v] = []
                
        inputs = set(dag.inputs) if hasattr(dag, 'inputs') else set(v for v in vertices if len(parents[v]) == 0)
        outputs = set(dag.outputs) if hasattr(dag, 'outputs') else None
        
        children = {v: [] for v in vertices}
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
                
        if outputs is None:
            outputs = set(v for v in vertices if len(children[v]) == 0)

    # --- Topological Sort ---
    # Use Kahn's algorithm
    in_degree = {v: len(parents[v]) for v in vertices}
    from collections import deque
    queue = deque()
    for v in vertices:
        if in_degree[v] == 0:
            queue.append(v)
            
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            temp_in_degree[c] -= 1
            if temp_in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(vertices):
        # Cycle or error, fallback to sorted
        topo_order = sorted(vertices)

    # --- Pebbling ---
    operations = []
    red_set = set()
    state = {v: 'not_loaded' for v in vertices} # not_loaded, red, blue(done)
    
    # To support recomputation efficiently, we need a function to ensure a vertex is red
    # This might trigger recursive loading/computing of parents
    
    import sys
    sys.setrecursionlimit(10000)
    
    def ensure_red(v):
        if state[v] == 'red':
            return
        if state[v] == 'blue':
            # If it's an output, it's stored. If we need it again, we must reload/recompute.
            # But outputs are terminal, so they shouldn't be needed again by other nodes.
            # However, if the DAG has shared subgraphs, an "output" of a subgraph might be an "input" to another part?
            # No, "output vertices" are the final targets.
            # If state is blue, it means we stored it. If we need it again, it must be an input or internal node that was stored?
            # The task says "all outputs must finish blue". It doesn't say internal nodes can be stored.
            # So internal nodes are never stored, only deleted or kept red.
            # If state[v] is 'blue', it implies v is an output and we are done with it.
            # If we call ensure_red on an output, something is wrong in the DAG structure (output used as parent).
            # We will assume DAG is a DAG.
            pass
            
        if v in inputs:
            operations.append(['load', v])
            state[v] = 'red'
            red_set.add(v)
        else:
            # Ensure parents are red
            for p in parents[v]:
                ensure_red(p)
            operations.append(['compute', v])
            state[v] = 'red'
            red_set.add(v)
            
    def delete_vertex(v):
        if state[v] == 'red':
            operations.append(['delete', v])
            state[v] = 'not_loaded' # Or 'deleted'
            red_set.discard(v)

    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
            
        # Ensure v is red
        ensure_red(v)
        
        # If v is an output, store it
        if v in outputs:
            operations.append(['store', v])
            state[v] = 'blue'
            red_set.discard(v)
        else:
            # Check memory budget
            if len(red_set) > fast_memory_budget:
                # Find a victim to delete
                # Heuristic: Delete the vertex with the fewest remaining children that are not yet processed
                # "Remaining children" = children of u that are in topo_order after current index
                # Precompute remaining children counts? Or just count on the fly.
                
                # To be fast, we can just pick the one with the smallest "future degree"
                # future_degree(u) = number of children c of u such that c has not been processed yet.
                
                # We can track a set of processed vertices.
                # Or simply: iterate through red_set, count children not in 'processed' set.
                
                # Optimization: Maintain a count of remaining children for each red vertex.
                # Let's use a simpler heuristic for now: Delete the vertex with the smallest index in topo_order (earliest)
                # This is a "FIFO" like strategy. It might cause recomputation but is simple.
                # Better: Delete the one with the fewest total children (static).
                
                victim = None
                min_children = float('inf')
                for u in red_set:
                    # Count children
                    c_count = len(children[u])
                    if c_count < min_children:
                        min_children = c_count
                        victim = u
                        
                if victim is not None and victim != v: # Don't delete the one we just computed
                    delete_vertex(victim)
                elif victim == v:
                    # If we must delete v, we just computed it. Deleting it immediately means we'll recompute it if needed.
                    # This is bad if it has many children. But if budget is 0 or 1, we have to.
                    delete_vertex(v)

    return operations
