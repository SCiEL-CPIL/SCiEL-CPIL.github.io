"""Communication-minimizing DAG pebbling with smart eviction strategy."""

def pebble(dag, fast_memory_budget):
    import time
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    # Parse the DAG structure
    if isinstance(dag, dict):
        n = dag.get('n', 0)
        if n == 0:
            # Try to infer from other fields
            vertices_list = dag.get('vertices', [])
            n = len(vertices_list)
        vertices = list(range(n))
        
        # Get parents for each vertex
        parents = {}
        if 'parents' in dag:
            for v in vertices:
                parents[v] = list(dag['parents'].get(v, []))
        elif 'edges' in dag:
            # edges might be list of (child, parent) or (parent, child)
            for edge in dag['edges']:
                if isinstance(edge, tuple) and len(edge) == 2:
                    # Assume (child, parent) or need to figure out direction
                    # Typically in pebbling, edges go from parent to child
                    # But let's assume the first element is the child
                    child, parent = edge
                    if child not in parents:
                        parents[child] = []
                    parents[child].append(parent)
        else:
            for v in vertices:
                parents[v] = []
        
        # Get inputs and outputs
        inputs = set()
        outputs = set()
        if 'inputs' in dag:
            inputs = set(dag['inputs'])
        elif 'sources' in dag:
            inputs = set(dag['sources'])
        if 'outputs' in dag:
            outputs = set(dag['outputs'])
        elif 'sinks' in dag:
            outputs = set(dag['sinks'])
        
        # If not specified, infer from structure
        if not inputs:
            inputs = set(v for v in vertices if not parents.get(v, []))
        if not outputs:
            # Outputs are vertices with no children
            children_map = {v: [] for v in vertices}
            for v in vertices:
                for p in parents.get(v, []):
                    if p in children_map:
                        children_map[p].append(v)
            outputs = set(v for v in vertices if not children_map.get(v, []))
    else:
        # Fallback
        vertices = list(range(len(dag)))
        parents = {}
        for i, item in enumerate(dag):
            if isinstance(item, dict):
                parents[i] = item.get('parents', [])
            elif isinstance(item, tuple) and len(item) >= 2:
                parents[i] = list(item[1])
            else:
                parents[i] = []
        inputs = set(v for v in vertices if not parents.get(v, []))
        children_map = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                if p in children_map:
                    children_map[p].append(v)
        outputs = set(v for v in vertices if not children_map.get(v, []))
    
    # Build children map
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children_map:
                children_map[p].append(v)
    
    # Compute remaining uses for each vertex
    # remaining_uses[v] = number of children that need v's value + 1 if v is an output
    remaining_uses = {}
    for v in vertices:
        uses = len(children_map.get(v, []))
        if v in outputs:
            uses += 1
        remaining_uses[v] = uses
    
    # Topological order
    topo_order = list(vertices)
    
    # Greedy pebbling with smart eviction
    fast_memory = set()  # Set of vertices currently in fast memory
    operations = []
    
    def evict_if_needed(exclude=None):
        """Evict vertices from fast memory if budget is exceeded"""
        exclude = exclude or set()
        while len(fast_memory) > fast_memory_budget:
            # Find vertex to evict
            candidates = [v for v in fast_memory if v not in exclude]
            if not candidates:
                candidates = list(fast_memory)
            
            # Priority: evict vertices with remaining_uses == 0 first
            # Then by fewest remaining uses
            # Then by earliest position in topo order (heuristic)
            
            def priority(v):
                ru = remaining_uses.get(v, 0)
                # Lower value = evict first
                if ru == 0:
                    return (0, 0, topo_order.index(v) if v in topo_order else 0)
                return (1, ru, topo_order.index(v) if v in topo_order else 0)
            
            candidates.sort(key=priority)
            evict_v = candidates[0]
            fast_memory.discard(evict_v)
            operations.append(('store', evict_v))
    
    # Process vertices in topological order
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        # Ensure all parents are in fast memory
        parent_set = set(parents.get(v, []))
        for p in parent_set:
            if p not in fast_memory:
                # Need to load p
                evict_if_needed(exclude=parent_set)
                fast_memory.add(p)
                operations.append(('load', p))
        
        # Compute v
        # Make room in fast memory for v
        evict_if_needed(exclude=parent_set)
        
        if len(fast_memory) < fast_memory_budget:
            fast_memory.add(v)
        else:
            # Force evict
            for evict_v in list(fast_memory):
                if evict_v not in parent_set:
                    fast_memory.discard(evict_v)
                    operations.append(('store', evict_v))
                    fast_memory.add(v)
                    break
            else:
                # All are parents, evict one (will reload later)
                evict_v = list(fast_memory)[0]
                fast_memory.discard(evict_v)
                operations.append(('store', evict_v))
                fast_memory.add(v)
        
        operations.append(('compute', v))
        
        # Decrement remaining_uses of all parents
        for p in parent_set:
            if p in remaining_uses:
                remaining_uses[p] -= 1
        
        # If v is an output, we need to ensure it's stored at the end
        # But for now, keep it in fast memory
    
    # Handle outputs: store all output vertices
    for v in outputs:
        if time.time() - start_time > DEADLINE:
            break
        if v in fast_memory:
            operations.append(('store', v))
            fast_memory.discard(v)
    
    # Truncate to max 20000 operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
