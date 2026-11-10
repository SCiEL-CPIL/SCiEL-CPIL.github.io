import time
import random
import itertools

DEADLINE = time.time() + 118

def _find_violation(adj, n):
    """Find one regular induced subgraph of size >= 6.
    Returns the subset as a tuple, or None if none found.
    Checks smaller sizes first as they are more likely and cheaper.
    """
    # Check k=6 first
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            # Create mask
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check degrees
            first_deg = (adj[subset[0]] & mask).bit_count()
            is_reg = True
            for v in subset[1:]:
                if (adj[v] & mask).bit_count() != first_deg:
                    is_reg = False
                    break
            
            if is_reg:
                return subset
    return None

def _min_conflicts_search(adj, n, max_iters=5000):
    """
    Min-Conflicts heuristic to eliminate regular induced subgraphs.
    adj: list of int bitmasks.
    """
    current_adj = adj[:]
    
    for _ in range(max_iters):
        if time.time() > DEADLINE - 2.0:
            break
            
        viol = _find_violation(current_adj, n)
        if viol is None:
            return current_adj, 0
        
        # Pick a random edge in the violating subset to flip
        # Or pick a random vertex in the subset and flip an edge incident to it
        # Flipping an edge within the subset is most likely to break regularity
        u_idx = random.randint(0, len(viol) - 1)
        v_idx = random.randint(0, len(viol) - 1)
        if u_idx == v_idx:
            continue
        u = viol[u_idx]
        v = viol[v_idx]
        
        # Flip edge (u, v)
        current_adj[u] ^= (1 << v)
        current_adj[v] ^= (1 << u)
        
    # If we exit loop, check final state
    final_viol = _find_violation(current_adj, n)
    return current_adj, 0 if final_viol is None else 1

def construct():
    start_time = time.time()
    best_n = 5
    best_edges = []
    
    # Try to find a graph for n=22, 21, 20...
    for n in range(22, 5, -1):
        if time.time() > DEADLINE - 10:
            break
            
        found = False
        # Number of restarts decreases as n increases to ensure we finish
        max_restarts = 50 if n <= 20 else 20
        
        for r in range(max_restarts):
            if time.time() > DEADLINE - 5:
                break
                
            # Initialize random graph
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
            # Run min-conflicts
            adj, viol_count = _min_conflicts_search(adj, n, max_iters=2000)
            
            if viol_count == 0:
                found = True
                # Convert to edges
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            edges.append([u, v])
                best_n = n
                best_edges = edges
                break
        
        if found:
            # We found a valid graph for this n, which is the max possible for this run
            # Since we go downwards, the first found is the best
            break
            
    return {'n': best_n, 'edges': best_edges}
