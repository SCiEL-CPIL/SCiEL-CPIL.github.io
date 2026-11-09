import time
import random
import math

N = 20
DEADLINE = 145.0  # Safety margin before 150s kill

def _popcount(x):
    return bin(x).count('1')

def _can_add_edge_fast(adj, u, v):
    """
    Check if adding edge (u,v) to a K3,3-free graph keeps it K3,3-free.
    adj is a list of int bitmasks.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Candidates for B-side (neighbors of u, excluding v)
    b_mask = nu & ~(1 << v)
    # Candidates for A-side (neighbors of v, excluding u)
    a_mask = nv & ~(1 << u)
    
    # Need at least 2 in each to form K3,3 with u,v
    if _popcount(b_mask) < 2 or _popcount(a_mask) < 2:
        return True
    
    # Convert masks to lists of vertices
    a_list = [i for i in range(N) if (a_mask >> i) & 1]
    b_list = [i for i in range(N) if (b_mask >> i) & 1]
    
    if len(a_list) < len(b_list):
        # Iterate pairs in a_list
        for i in range(len(a_list)):
            a1 = a_list[i]
            adj_a1 = adj[a1]
            for j in range(i + 1, len(a_list)):
                a2 = a_list[j]
                adj_a2 = adj[a2]
                # Common neighbors of a1 and a2 within b_mask
                common = adj_a1 & adj_a2 & b_mask
                if _popcount(common) >= 2:
                    return False
    else:
        # Iterate pairs in b_list
        for i in range(len(b_list)):
            b1 = b_list[i]
            adj_b1 = adj[b1]
            for j in range(i + 1, len(b_list)):
                b2 = b_list[j]
                adj_b2 = adj[b2]
                # Common neighbors of b1 and b2 within a_mask
                common = adj_b1 & adj_b2 & a_mask
                if _popcount(common) >= 2:
                    return False
    return True

def _edge_count(adj):
    return sum(_popcount(adj[u]) for u in range(N)) // 2

def _greedy_fill(adj, time_limit):
    """
    Greedily add all possible edges.
    """
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if time.time() - start_time > time_limit:
                    return adj
                if (adj[u] >> v) & 1:
                    continue
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
    return adj

def _local_search_polish(adj, time_limit):
    """
    Fast local search: try removing one edge and adding one or two edges.
    Used as a final polish after Simulated Annealing.
    """
    start_time = time.time()
    current_count = _edge_count(adj)
    best_adj = adj[:]
    best_count = current_count
    
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    while time.time() - start_time < time_limit:
        improved = False
        
        missing = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]
        existing = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]
        
        # Try 1-remove, 2-add swap (net +1)
        for u, v in existing:
            if time.time() - start_time > time_limit:
                return best_adj
            
            # Remove edge (u,v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            added = 0
            new_edges = []
            
            # Try to add up to 2 missing edges
            for u2, v2 in missing:
                if added >= 2:
                    break
                if (adj[u2] >> v2) & 1:
                    continue
                if _can_add_edge_fast(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    added += 1
                    new_edges.append((u2, v2))
            
            if added >= 2:
                current_count += (added - 1)
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Revert
                for eu, ev in new_edges:
                    adj[eu] &= ~(1 << ev)
                    adj[ev] &= ~(1 << eu)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        if not improved:
            break
            
    return best_adj

def _algebraic_start():
    """
    Generate a dense, symmetric starting graph using a structured approach.
    """
    adj = [0] * N
    
    # 5 blocks of 4 vertices each
    for i in range(5):
        for j in range(5):
            if i == j:
                # Within block: use a cycle
                for a in range(4):
                    b = (a + 1) % 4
                    u = i * 4 + a
                    v = i * 4 + b
                    if u < v:
                        if _can_add_edge_fast(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
            else:
                offset = (j - i) % 5
                for a in range(4):
                    for b in range(4):
                        u = i * 4 + a
                        v = j * 4 + b
                        if u >= v:
                            continue
                        if offset in (1, 4):
                            if (b - a) % 4 in (0, 1):
                                if _can_add_edge_fast(adj, u, v):
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                        elif offset in (2, 3):
                            if (b - a) % 4 == 0:
                                if _can_add_edge_fast(adj, u, v):
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
    return adj

def _random_start(density):
    """Generate a random graph with given edge probability, filtered for K3,3."""
    adj = [0] * N
    for u in range(N):
        for v in range(u + 1, N):
            if random.random() < density:
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    return adj

def _simulated_annealing(adj, time_limit):
    """
    Simulated Annealing to optimize the number of edges in a K3,3-free graph.
    """
    start_time = time.time()
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Initial temperature
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995  # Adjust based on expected iterations
    
    current_count = _edge_count(adj)
    best_adj = adj[:]
    best_count = current_count
    
    # Precompute list of existing and missing edges for faster access if possible, 
    # but since the graph changes, we might need to update dynamically or just iterate.
    # For N=20, 190 pairs is small, so iterating all pairs is fast enough.
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a random move
        # Types:
        # 1. Add a random missing edge
        # 2. Remove a random existing edge
        # 3. Swap: Remove one existing, try to add one or two missing
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random missing edge
            # Find a missing edge
            missing_edges = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]
            if not missing_edges:
                continue
            u, v = random.choice(missing_edges)
            
            if _can_add_edge_fast(adj, u, v):
                # Accept
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Reject (energy increase is infinite/forbidden)
                pass
                
        elif move_type < 0.7:
            # Try to remove a random existing edge
            existing_edges = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]
            if not existing_edges:
                continue
            u, v = random.choice(existing_edges)
            
            # Remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_count -= 1
            
            # Acceptance check
            # Delta E = +1 (since E = -count)
            # Prob = exp(-1/T)
            if random.random() < math.exp(-1.0 / T):
                # Accept
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_count += 1
                
        else:
            # Swap: Remove one existing, try to add up to 2 missing
            existing_edges = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]
            missing_edges = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]
            
            if not existing_edges or not missing_edges:
                continue
                
            u_rem, v_rem = random.choice(existing_edges)
            
            # Remove
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_count -= 1
            
            added = 0
            new_edges = []
            
            # Try to add 2 random missing edges
            # Shuffle missing to pick random ones
            random.shuffle(missing_edges)
            for u2, v2 in missing_edges:
                if added >= 2:
                    break
                if (adj[u2] >> v2) & 1:
                    continue
                if _can_add_edge_fast(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    added += 1
                    new_edges.append((u2, v2))
            
            delta_count = added - 1
            new_count = current_count + delta_count
            
            # Acceptance
            if delta_count > 0:
                # Always accept improvement
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Delta E = -delta_count (since E = -count, delta E = -delta_count)
                # If delta_count is -1, delta E = 1. Prob = exp(-1/T)
                # If delta_count is 0, delta E = 0. Prob = 1.
                if delta_count == 0:
                    # Accept
                    current_count = new_count
                else:
                    prob = math.exp(-(-delta_count) / T) # -(-delta_count) = delta_count which is negative, so -delta_count is positive
                    # Wait, E = -count. Delta E = E_new - E_old = -count_new - (-count_old) = count_old - count_new = -delta_count.
                    # If delta_count = -1, Delta E = 1. Prob = exp(-1/T).
                    # If delta_count = -2, Delta E = 2. Prob = exp(-2/T).
                    if random.random() < prob:
                        current_count = new_count
                    else:
                        # Revert
                        for eu, ev in new_edges:
                            adj[eu] &= ~(1 << ev)
                            adj[ev] &= ~(1 << eu)
                        adj[u_rem] |= (1 << v_rem)
                        adj[v_rem] |= (1 << u_rem)
                        current_count += 1 # Revert the removal
            
            # Note: In the 'else' branch of swap, if we revert, we restore state.
            # If we accept, we keep the new state.
            # The code above handles this.
            
        T *= cooling_rate

    return best_adj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_adj = [0] * N
    best_count = 0
    
    # Strategy 1: Algebraic construction + Greedy + SA
    adj = _algebraic_start()
    adj = _greedy_fill(adj, min(2.0, deadline - time.time() - 10))
    current_count = _edge_count(adj)
    if current_count > best_count:
        best_adj = adj[:]
        best_count = current_count
    
    # Run SA on the greedy-filled algebraic graph
    remaining = deadline - time.time() - 5
    if remaining > 2:
        adj = best_adj[:]
        adj = _simulated_annealing(adj, remaining * 0.6)
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
        
        # Polish
        remaining = deadline - time.time() - 2
        if remaining > 1:
            adj = _local_search_polish(adj, remaining)
            current_count = _edge_count(adj)
            if current_count > best_count:
                best_adj = adj[:]
                best_count = current_count

    # Strategy 2: Random starts with SA
    densities = [0.5, 0.6, 0.7, 0.8, 0.4]
    random.shuffle(densities)
    
    for density in densities:
        if time.time() > deadline - 15:
            break
            
        adj = _random_start(density)
        adj = _greedy_fill(adj, min(1.0, deadline - time.time() - 10))
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
            
        remaining = deadline - time.time() - 5
        if remaining < 2:
            break
            
        adj = _simulated_annealing(adj, remaining * 0.7)
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
            
        remaining = deadline - time.time() - 2
        if remaining > 1:
            adj = _local_search_polish(adj, remaining)
            current_count = _edge_count(adj)
            if current_count > best_count:
                best_adj = adj[:]
                best_count = current_count

    # Final polish on the global best
    remaining = deadline - time.time() - 1
    if remaining > 2:
        adj = best_adj[:]
        adj = _local_search_polish(adj, remaining)
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count

    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
