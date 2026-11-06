import time
import random
import math

N = 20
DEADLINE_MARGIN = 5.0

def _bit_count(x):
    return x.bit_count()

def _can_add(adj, u, v):
    """
    Check if edge (u, v) can be added to adj without creating K2,3.
    adj is a list of ints, where adj[i] is a bitmask of neighbors.
    """
    # 1. Direct common neighbors of u and v must be <= 2
    if _bit_count(adj[u] & adj[v]) > 2:
        return False
    
    # 2. For any w in N(v), w != u:
    #    Adding (u,v) adds v to N(u).
    #    If v is in N(w), then v becomes a common neighbor of u and w.
    #    So we need current common neighbors of u and w to be <= 1.
    mask_v = adj[v]
    # Iterate over set bits in mask_v
    # We can just loop w from 0 to N-1 for simplicity since N=20 is small
    for w in range(N):
        if w == u or w == v:
            continue
        if (mask_v >> w) & 1:
            if _bit_count(adj[u] & adj[w]) > 1:
                return False
                
    # 3. For any w in N(u), w != v:
    #    Adding (u,v) adds u to N(v).
    #    If u is in N(w), then u becomes a common neighbor of v and w.
    #    So we need current common neighbors of v and w to be <= 1.
    mask_u = adj[u]
    for w in range(N):
        if w == u or w == v:
            continue
        if (mask_u >> w) & 1:
            if _bit_count(adj[v] & adj[w]) > 1:
                return False
                
    return True

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _count_edges(adj):
    # Sum bits in all masks, divide by 2
    total = 0
    for i in range(N):
        total += _bit_count(adj[i])
    return total // 2

def _greedy_init(seed):
    rng = random.Random(seed)
    adj = [0] * N
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if _can_add(adj, u, v):
            _add_edge(adj, u, v)
            
    return adj, _count_edges(adj)

def _simulated_annealing(adj_init, edges_init, seed, time_limit):
    rng = random.Random(seed)
    adj = adj_init[:]
    edges = edges_init
    best_adj = adj[:]
    best_edges = edges
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Initial temperature
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    while time.time() < deadline:
        # Choose a move
        # 60% Add, 20% Remove, 20% Swap
        r = rng.random()
        
        if r < 0.6:
            # Try to add a random missing edge
            # Find a random missing edge
            # For efficiency, we can just pick random u, v and check if missing
            # But we need to ensure we try enough candidates
            attempts = 0
            added = False
            while attempts < 20:
                u = rng.randint(0, N-1)
                v = rng.randint(0, N-1)
                if u == v:
                    attempts += 1
                    continue
                if u > v:
                    u, v = v, u
                # Check if missing
                if not ((adj[u] >> v) & 1):
                    if _can_add(adj, u, v):
                        _add_edge(adj, u, v)
                        edges += 1
                        added = True
                        if edges > best_edges:
                            best_edges = edges
                            best_adj = adj[:]
                        break
                attempts += 1
            if not added:
                # If we failed to add, we might just continue
                pass
                
        elif r < 0.8:
            # Try to remove a random existing edge
            # Pick random u, v that are connected
            attempts = 0
            removed = False
            while attempts < 20:
                u = rng.randint(0, N-1)
                v = rng.randint(0, N-1)
                if u == v:
                    attempts += 1
                    continue
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    # Delta E = +1 (worse)
                    # Accept with prob exp(1/T)
                    if rng.random() < math.exp(1.0 / T):
                        _remove_edge(adj, u, v)
                        edges -= 1
                        removed = True
                        break
                attempts += 1
            if not removed:
                pass
        else:
            # Swap: Remove one, Add one
            # Pick random existing edge to remove
            attempts_rm = 0
            u_rm, v_rm = -1, -1
            while attempts_rm < 20:
                u = rng.randint(0, N-1)
                v = rng.randint(0, N-1)
                if u == v:
                    attempts_rm += 1
                    continue
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    u_rm, v_rm = u, v
                    break
                attempts_rm += 1
            
            if u_rm != -1:
                # Remove it temporarily
                _remove_edge(adj, u_rm, v_rm)
                edges -= 1
                
                # Try to add a random missing edge
                attempts_add = 0
                added = False
                while attempts_add < 20:
                    u = rng.randint(0, N-1)
                    v = rng.randint(0, N-1)
                    if u == v or (u == u_rm and v == v_rm) or (u == v_rm and v == u_rm):
                        attempts_add += 1
                        continue
                    if u > v:
                        u, v = v, u
                    if not ((adj[u] >> v) & 1):
                        if _can_add(adj, u, v):
                            _add_edge(adj, u, v)
                            edges += 1
                            added = True
                            if edges > best_edges:
                                best_edges = edges
                                best_adj = adj[:]
                            break
                    attempts_add += 1
                
                if not added:
                    # Revert removal
                    _add_edge(adj, u_rm, v_rm)
                    edges += 1
                    # If we reverted, we are back to original state.
                    # Maybe accept this "swap" as a neutral move if it changed structure?
                    # But it didn't change structure because we reverted.
                    # So this was a failed swap.
                    pass

        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: Reheat or restart
            # For now, keep T at min and continue exploring
            # Or restart SA with new init?
            # Let's just continue with low T (greedy)
            
    return best_adj, best_edges

def construct():
    global deadline
    deadline = time.time() + 150.0
    
    best_adj = None
    best_edges = 0
    
    # Run multiple SA restarts
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616]
    
    for seed in seeds:
        if time.time() > deadline - DEADLINE_MARGIN:
            break
            
        # Greedy init
        adj, edges = _greedy_init(seed)
        
        # SA
        adj_sa, edges_sa = _simulated_annealing(adj, edges, seed, 5.0) # 5s per restart
        
        if edges_sa > best_edges:
            best_edges = edges_sa
            best_adj = adj_sa
            
    if best_adj is None:
        # Fallback
        adj = [0] * N
        best_adj = adj
        best_edges = 0
        
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
