"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _creates_c4(adj, u, v):
    """
    Check if adding edge (u, v) to the graph with adjacency sets 'adj' creates a C4.
    A C4 is u-x-y-v-u.
    This exists if there is an edge (x, y) such that x in adj[u] and y in adj[v].
    We iterate x in adj[u] and check if any neighbor of x is in adj[v].
    """
    nu = adj[u]
    nv = adj[v]
    for x in nu:
        # Check if any neighbor of x is in nv
        # adj[x] is the set of neighbors of x
        if adj[x] & nv:
            return True
    return False

def _greedy_construct(n, min_deg_required, time_limit=5.0):
    """
    Attempt to construct a C4-free graph on n vertices with min degree >= min_deg_required.
    Uses a greedy strategy: repeatedly add an edge that does not create a C4,
    prioritizing edges that help low-degree vertices.
    """
    start_time = time.time()
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # We will try multiple random orders of edges
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
            
        adj = [set() for _ in range(n)]
        
        # Shuffle edges to randomize the greedy choice
        random.shuffle(all_edges)
        
        # To make it faster, we can just iterate through the shuffled list.
        # However, a simple single pass might not reach the max degree if we are not careful.
        # A better greedy: 
        # While there exists an edge (u,v) not in graph such that adding it doesn't create C4:
        #   Pick one (e.g., the one with lowest max degree of endpoints, or random)
        #   Add it.
        
        # Since checking all edges is O(E * C4_check), and E ~ n^2/2, this is O(n^4) roughly.
        # n=47, n^4 ~ 4.8M. C4 check is fast. 500 attempts might be too slow.
        # Let's optimize: 
        # We can just iterate through the shuffled list and add if valid.
        # This is a "random greedy" or "online" algorithm.
        # It might get stuck with low degrees.
        # To improve, we can do multiple passes over the edges?
        # Or just accept the result of one pass and check degree.
        
        for u, v in all_edges:
            if time.time() - start_time > time_limit:
                return None # Timeout, return None to try next n or fail
            
            # Skip if already added
            if v in adj[u]:
                continue
                
            # Check C4
            if _creates_c4(adj, u, v):
                continue
                
            # Heuristic: Only add if it helps balance? 
            # For now, just add. Random order helps.
            adj[u].add(v)
            adj[v].add(u)
            
        # Check min degree
        min_deg = min(len(adj[i]) for i in range(n))
        
        if min_deg >= min_deg_required:
            # Success
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
            
    return None

def _repair_graph(n, adj, min_deg_required, time_limit=5.0):
    """
    Try to repair a graph that is C4-free but has min degree < min_deg_required.
    We look for vertices with low degree and try to add edges to them.
    If adding an edge creates a C4, we try to remove an edge from the C4 that 
    doesn't drop any degree below min_deg_required (or below current min if we are improving).
    This is complex. 
    Alternative: Just return None and rely on more random attempts in _greedy_construct.
    Given the low min degree requirement (7 or 8), random greedy should work well.
    """
    pass

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin before 120s limit
    
    # Try from largest n down
    # n=47 requires min degree 8.
    # n=46 requires min degree 7.
    # n=45 requires min degree 6.
    # etc.
    
    # Allocate time based on n. Larger n is harder? 
    # Actually, smaller min degree is easier. 
    # n=47 (deg 8) is harder than n=46 (deg 7).
    # So spend more time on n=47.
    
    time_budgets = {
        47: 30.0,
        46: 30.0,
        45: 20.0,
        44: 10.0,
        43: 5.0,
        42: 5.0,
        41: 5.0,
        40: 5.0,
        39: 2.0
    }
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
            
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 2:
            break
            
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        # Cap the time for this n
        t_limit = min(time_budgets.get(n, 5.0), remaining_time - 1)
        
        if t_limit <= 0:
            continue
            
        result = _greedy_construct(n, min_deg_req, time_limit=t_limit)
        if result is not None:
            return result
            
    # Fallback: Construct a known small C4-free graph
    # For n=39, min degree 0 is fine.
    # But we can do better.
    # Let's just return a valid small graph if we fail.
    # The task says "Maximize n". If we fail 46, we should try to return the best we found.
    # But _greedy_construct returns None on failure.
    # We should keep track of the best result.
    
    # Let's re-structure to keep best result.
    # Actually, the loop above returns immediately on success.
    # If it fails all, we return a default.
    
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
