import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # ... (graph extraction and topo sort) ...
    
    # If topo sort takes too long, fallback to simple heuristic?
    # Topo sort is O(N+E), should be fast.
    
    # ... (simulation) ...
    
    for v in topo_order:
        if time.time() > deadline:
            # Return best effort
            break
        # ...
        
    # Ensure outputs are stored
    for v in outputs:
        if time.time() > deadline:
            break
        if v in red:
            ops.append(('store', v))
            red.remove(v)
            blue.add(v)
            
    return ops
