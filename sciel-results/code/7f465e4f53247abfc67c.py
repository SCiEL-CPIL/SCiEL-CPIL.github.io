import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 55.0  # 5s safety margin
    
    # Parse DAG
    if isinstance(dag, dict):
        parents = dag.get('parents', [])
        outputs = list(dag.get('outputs', []))
        num_vertices = dag.get('num_vertices', len(parents))
    else:
        parents = list(dag)
        outputs = []
        num_vertices = len(parents)
    
    # Compute children
    children = [[] for _ in range(num_vertices)]
    for v in range(num_vertices):
        for p in parents[v]:
            children[p].append(v)
    
    # Inputs: no parents
    inputs = [v for v in range(num_vertices) if len(parents[v]) == 0]
    
    # If no outputs specified, use vertices with no children
    if not outputs:
        outputs = [v for v in range(num_vertices) if len(children[v]) == 0]
    
    output_set = set(outputs)
    
    # State tracking
    in_fast = set(inputs)  # Inputs start blue
    computed = set(inputs)
    
    operations = []
    
    # Precompute: for each vertex, the list of children that come after it
    # and the minimum distance to next child
    next_child_dist = [0] * num_vertices
    for v in range(num_vertices):
        future_children = [c for c in children[v] if c > v]
        if future_children:
            next_child_dist[v] = min(future_children) - v
        else:
            next_child_dist[v] = float('inf')
    
    # Process in topological order
    for v in range(num_vertices):
        if time.time() - start_time > DEADLINE:
            break
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                if p in computed:
                    operations.append(['load', p])
                    in_fast.add(p)
                else:
                    # Recompute: this shouldn't happen in topo order, but handle it
                    pass
        
        # Compute v
        operations.append(['compute', v])
        computed.add(v)
        in_fast.add(v)
        
        # Manage memory
        while len(in_fast) > fast_memory_budget:
            # Find vertex to evict
            best_evict = None
            best_score = float('inf')
            
            for u in in_fast:
                if u in output_set:
                    continue
                
                # Will u be needed again?
                future_children = [c for c in children[u] if c > v]
                
                if not future_children:
                    score = 0  # Safe to delete
                else:
                    # Score: lower = evict first
                    # Prefer evicting vertices with far next need and low fan-out
                    min_dist = min(future_children) - v
                    score = 1000.0 / (1 + min_dist) + len(future_children) * 5
                
                if score < best_score:
                    best_score = score
                    best_evict = u
            
            if best_evict is None:
                break
            
            # Decide: store or delete
            future_children = [c for c in children[best_evict] if c > v]
            if future_children:
                operations.append(['store', best_evict])
            else:
                operations.append(['delete', best_evict])
            
            in_fast.remove(best_evict)
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in in_fast:
            operations.append(['load', v])
            in_fast.add(v)
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
