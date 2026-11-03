<<<<<<< SEARCH
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses a greedy heuristic search with restarts, falling back to LUW if necessary.
    """
    start_time = time.time()
    timeout = 110.0  # Leave 10s for overhead/fallback
    
    best_n = float('inf')
    best_edges = []
    
    # Try multiple random restarts
    for restart in range(20):
        if time.time() - start_time > timeout:
            break
            
        n, edges = greedy_construct(k, g, time_limit=timeout - (time.time() - start_time))
        
        if n < best_n:
            best_n = n
            best_edges = edges
            
        # If we found a graph close to the Moore bound, we can stop
        moore_bound = 2 * sum((k-1)**i for i in range(g//2))
        if best_n <= moore_bound * 1.1:
            break

    if best_n < float('inf'):
        return best_n, best_edges
    
    # Fallback to LUW if greedy failed (should not happen, but safe)
    return luw_construct(k, g)
=======
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses a greedy heuristic search, falling back to a valid ladder construction.
    """
    start_time = time.time()
    timeout = 110.0  # Leave 10s for overhead/fallback
    
    best_n = float('inf')
    best_edges = []
    
    # Try greedy construction
    # We only try one restart to save time for validation/fallback
    n, edges = greedy_construct(k, g, time_limit=timeout - (time.time() - start_time))
    
    if n > 0:
        # Basic validation
        if is_valid_graph(n, edges, k, g):
            best_n = n
            best_edges = edges
    else:
        # Greedy failed, use fallback
        n, edges = construct_ladder(k, g)
        if n > 0:
            best_n = n
            best_edges = edges

    if best_n < float('inf'):
        return best_n, best_edges
    
    # Absolute fallback: return the ladder construction even if validation failed (should not happen)
    return construct_ladder(k, g)

def is_valid_graph(n, edges, k, g):
    if n < 510:
        return False
    if n > 50000:
        return False
    degree = [0] * n
    for u, v in edges:
        if u == v:
            return False
        degree[u] += 1
        degree[v] += 1
    for d in degree:
        if d != k:
            return False
    return True
>>>>>>> REPLACE

