<<<<<<< SEARCH
def local_search_optimized(n, adj, deadline):
    """Optimized local search to minimize C5 count and maximize density."""
    best_adj = adj[:]
    best_cost = count_c5_fast(n, adj)
    
    T = 50.0
    T_min = 0.1
    cooling = 0.99
    
    while T > T_min and time.time() < deadline:
        # Pick random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        curr_cost = best_cost
        
        # Flip edge
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        new_cost = count_c5_fast(n, adj)
        
        # Accept if better or with probability
        if new_cost < curr_cost or (T > 0 and random.random() < np.exp(min(50, (curr_cost - new_cost) / T))):
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = adj[:]
            if best_cost == 0:
                # Check alpha heuristic to see if we should stop
                if calc_alpha_heuristic(n, adj) < 9:
                    break
        else:
            # Revert
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        T *= cooling
    
    return best_adj
=======
def local_search_optimized(n, adj, deadline):
    """Optimized local search: minimize C5 count, then minimize alpha."""
    best_adj = adj[:]
    best_c5 = count_c5_fast(n, adj)
    best_alpha = calc_alpha_heuristic(n, adj) if best_c5 == 0 else 99
    
    T = 30.0
    T_min = 0.05
    cooling = 0.995
    
    step = 0
    while T > T_min and time.time() < deadline:
        step += 1
        # Pick random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        # Flip edge
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        new_c5 = count_c5_fast(n, adj)
        
        if new_c5 == 0:
            new_alpha = calc_alpha_heuristic(n, adj)
            # Cost: prioritize C5=0, then low alpha
            curr_cost = best_c5 * 100 + best_alpha
            new_cost = new_c5 * 100 + new_alpha
        else:
            curr_cost = best_c5 * 100 + best_alpha
            new_cost = new_c5 * 100 + 99
        
        # Accept if better or with probability
        accept = False
        if new_cost < curr_cost:
            accept = True
        elif T > 0 and random.random() < np.exp(min(50, (curr_cost - new_cost) / T)):
            accept = True
        
        if accept:
            if new_cost < best_c5 * 100 + best_alpha:
                best_c5 = new_c5
                best_alpha = new_alpha if new_c5 == 0 else 99
                best_adj = adj[:]
            if best_c5 == 0 and best_alpha < 9:
                break
        else:
            # Revert
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        T *= cooling
    
    return best_adj
>>>>>>> REPLACE
