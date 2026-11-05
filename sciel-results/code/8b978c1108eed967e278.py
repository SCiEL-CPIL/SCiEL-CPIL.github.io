<<<<<<< SEARCH
def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # 1. Try LUW Graph
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. If k=4, g=10, try to find a smaller graph
    if k == 4 and g == 10:
        target_n = 320
        rng = random.Random(42)
        
        # Try multiple random starts with simulated annealing
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            edges = _generate_4_regular(target_n, seed=1000 + attempt * 7)
            if edges is None:
                continue
                
            # Check if it's already valid (very unlikely)
            if _validate_graph(target_n, edges, k, g):
                best_n = target_n
                best_edges = edges
                break
                
            # Try to improve girth
            time_left = deadline - time.time()
            if time_left < 3:
                break
                
            improved_edges, cost = _improve_girth(target_n, edges, g, time_left - 1, rng)
            
            if cost == 0:
                if _validate_graph(target_n, improved_edges, k, g):
                    best_n = target_n
                    best_edges = improved_edges
                    break
                    
    return best_n, best_edges
=======
def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # 1. Try LUW Graph
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. Optional: Try to find a smaller graph if time permits
    # For now, we rely on the improved LUW construction which is deterministic and fast.
    # The random search is too slow and unlikely to beat the LUW bound for g=10.
    
    return best_n, best_edges
>>>>>>> REPLACE
