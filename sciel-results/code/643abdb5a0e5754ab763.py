<<<<<<< SEARCH
def construct():
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges, n)
    
    # Deadline guard
    deadline = time.time() + 80  # 80s safety margin
    
    used_edges = set()
    cycles = []
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    for attempt in range(3):
        if time.time() > deadline:
            break
        
        cycle = _find_hamiltonian_cycle(adj, used_edges, n)
        if cycle is None:
            break
        
        # Add all edges of this cycle to used_edges
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add(_edge_key(u, v))
        
        cycles.append(cycle)
    
    # Ensure at least one cycle; if none found, return a single Hamiltonian cycle
    if not cycles:
        # Fallback: find one without edge constraints
        used_edges = set()
        cycle = _find_hamiltonian_cycle(adj, used_edges, n)
        if cycle:
            cycles = [cycle]
        else:
            # Absolute fallback
            cycles = [list(range(n))]
    
    # Convert to proper types
    result = []
    for c in cycles:
        result.append(tuple(int(v) for v in c))
    
    return result
=======
def construct():
    import random
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges, n)
    
    # Deadline guard with safety margin
    overall_deadline = time.time() + 85
    
    used_edges = set()
    cycles = []
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    for attempt in range(3):
        if time.time() > overall_deadline:
            break
        
        # Try multiple starting vertices with randomization for robustness
        found = False
        starts = list(range(n))
        random.shuffle(starts)
        
        # First try start=0 (deterministic), then others
        if attempt == 0:
            ordered_starts = [0] + [s for s in starts if s != 0]
        else:
            ordered_starts = starts
        
        for start in ordered_starts:
            if time.time() > overall_deadline:
                break
            
            remaining_time = max(1.0, overall_deadline - time.time() - 5.0)
            cycle = _find_hamiltonian_cycle(adj, used_edges, n, start=start, time_limit=remaining_time)
            if cycle is not None:
                # Add all edges of this cycle to used_edges
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    used_edges.add(_edge_key(u, v))
                cycles.append(cycle)
                found = True
                break
        
        if not found:
            break
    
    # Ensure at least one cycle; if none found, return a single Hamiltonian cycle
    if not cycles:
        # Fallback: find one without edge constraints
        used_edges = set()
        for start in range(n):
            if time.time() > overall_deadline:
                break
            cycle = _find_hamiltonian_cycle(adj, used_edges, n, start=start, time_limit=max(1.0, overall_deadline - time.time() - 2.0))
            if cycle:
                cycles = [cycle]
                break
        if not cycles:
            # Absolute fallback
            cycles = [list(range(n))]
    
    # Convert to proper types
    result = []
    for c in cycles:
        result.append(tuple(int(v) for v in c))
    
    return result
>>>>>>> REPLACE
