def pebble(dag, fast_memory_budget):
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # remaining_uses[v] = number of children of v not yet processed
    remaining_uses = {v: len(children[v]) for v in vertices}
    
    in_fast = set()   # vertices in fast memory (red)
    in_blue = set(inputs)  # vertices in blue (slow memory)
    
    operations = []
    
    for v in vertices:
        ps = parents.get(v, [])
        
        # Load parents into fast memory
        for p in ps:
            if p not in in_fast:
                if p in in_blue:
                    operations.append(['load', p])
                    in_blue.discard(p)
                    in_fast.add(p)
                else:
                    # Should not happen, but skip
                    pass
        
        # Compute v
        operations.append(['compute', v])
        in_fast.add(v)
        
        # Update remaining_uses for parents
        for p in ps:
            if p in remaining_uses:
                remaining_uses[p] -= 1
        
        # Free space: delete vertices with no remaining uses (non-output)
        to_delete = [u for u in in_fast 
                     if u not in outputs and remaining_uses.get(u, 0) == 0]
        for u in to_delete:
            operations.append(['delete', u])
            in_fast.discard(u)
        
        # If over budget, store vertices with lowest remaining_uses
        while len(in_fast) > fast_memory_budget:
            candidates = [u for u in in_fast if remaining_uses.get(u, 0) > 0]
            if not candidates:
                candidates = list(in_fast)
            if not candidates:
                break
            u_to_store = min(candidates, key=lambda u: remaining_uses.get(u, 0))
            operations.append(['store', u_to_store])
            in_fast.discard(u_to_store)
            in_blue.add(u_to_store)
    
    # Ensure outputs are in blue
    for v in outputs:
        if v in in_fast:
            operations.append(['store', v])
            in_fast.discard(v)
            in_blue.add(v)
    
    return operations
