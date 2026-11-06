import time
import random

N = 20
DEADLINE_FRACTION = 0.95

def _can_add_edge(adj, u, v):
    """
    Check if edge (u, v) can be added to the graph represented by adj (list of bitmasks).
    Returns True if valid.
    """
    if u == v:
        return False
    if adj[u] & (1 << v):
        return False

    nu = adj[u]
    nv = adj[v]
    
    # Condition 1: For all x, if v in N(x), then |N(u) cap N(x)| <= 1
    # We iterate x from 0 to N-1.
    # If v is in N(x), i.e., adj[x] has bit v set.
    for x in range(N):
        if x == u or x == v:
            continue
        if adj[x] & (1 << v):
            # v is a neighbor of x
            # Check common neighbors of u and x
            # Before adding (u,v), v is not in N(u).
            # So N(u) cap N(x) does not contain v.
            # After adding (u,v), v will be in N(u).
            # So new common neighbors = (N(u) cap N(x)) union {v} (if v in N(x))
            # We need size <= 2.
            # Current size = (nu & adj[x]).bit_count()
            # If v in N(x), new size = current size + 1.
            # So we need current size <= 1.
            if (nu & adj[x]).bit_count() > 1:
                return False

    # Condition 2: For all x, if u in N(x), then |N(v) cap N(x)| <= 1
    for x in range(N):
        if x == u or x == v:
            continue
        if adj[x] & (1 << u):
            # u is a neighbor of x
            # Before adding (u,v), u is not in N(v).
            # So N(v) cap N(x) does not contain u.
            # After adding (u,v), u will be in N(v).
            # New common neighbors = (N(v) cap N(x)) union {u}
            # We need size <= 2.
            # Current size = (nv & adj[x]).bit_count()
            # So we need current size <= 1.
            if (nv & adj[x]).bit_count() > 1:
                return False

    return True

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _edge_count(adj):
    # Sum of bit counts divided by 2
    total = 0
    for mask in adj:
        total += mask.bit_count()
    return total // 2

def _greedy_construct(seed):
    """
    Construct an initial K3,3-free graph using greedy edge addition.
    """
    random.seed(seed)
    adj = [0] * N
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if _can_add_edge(adj, u, v):
            _add_edge(adj, u, v)
            
    return adj

def _local_search(adj, deadline):
    """
    Perform local search (edge swaps) to improve the graph.
    """
    best_adj = adj[:]
    best_count = _edge_count(adj)
    current_adj = adj[:]
    current_count = best_count
    
    # Pre-compute list of all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    while time.time() < deadline:
        # Try to add any missing edge
        improved = False
        missing_edges = [(u, v) for u, v in all_edges if not (current_adj[u] & (1 << v))]
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if _can_add_edge(current_adj, u, v):
                _add_edge(current_adj, u, v)
                current_count += 1
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
        
        if improved:
            continue # Restart the loop to check for more additions
        
        # If no single edge can be added, try a swap
        # Pick a random existing edge to remove
        existing_edges = [(u, v) for u, v in all_edges if (current_adj[u] & (1 << v))]
        if not existing_edges:
            break
            
        random.shuffle(existing_edges)
        swap_found = False
        
        for u1, v1 in existing_edges:
            # Try to find a missing edge to add
            random.shuffle(missing_edges)
            for u2, v2 in missing_edges:
                # Don't swap with the same edge
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                    continue
                
                # Remove u1-v1
                _remove_edge(current_adj, u1, v1)
                current_count -= 1
                
                # Try to add u2-v2
                if _can_add_edge(current_adj, u2, v2):
                    _add_edge(current_adj, u2, v2)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = current_adj[:]
                    swap_found = True
                    break
                else:
                    # Revert removal
                    _add_edge(current_adj, u1, v1)
                    current_count += 1
            
            if swap_found:
                break
        
        if not swap_found:
            # If no simple swap works, we might be stuck in a deep local optimum.
            # For now, we just continue trying different random swaps.
            # A more aggressive move would be 2-swaps, but let's see if 1-swaps suffice.
            pass

    return best_adj, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION
    
    best_adj = None
    best_count = 0
    
    # Try multiple random restarts
    seed = 42
    while time.time() < deadline:
        seed += 1
        # Greedy construction
        adj = _greedy_construct(seed)
        
        # Local search
        adj, count = _local_search(adj, deadline)
        
        if count > best_count:
            best_count = count
            best_adj = adj
        
        # If we have a very high count, we can be more conservative with time?
        # No, just keep going.
        
        # Safety check: if time is running out, break
        if time.time() > deadline:
            break

    if best_adj is None:
        # Fallback: empty graph
        best_adj = [0] * N
        best_count = 0

    # Convert bitmask adj to list of edges
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
