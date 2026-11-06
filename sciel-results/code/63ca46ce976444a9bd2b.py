import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(all_edges)
    
    def is_valid_add(adj, u, v):
        # adj is a list of integers (bitmasks)
        # 1. Check common neighbors of u and v
        # If they already have > 2 common neighbors, adding the edge doesn't change that, 
        # but the graph is already invalid? No, we maintain validity.
        # Actually, if the graph is currently valid, N(u) and N(v) have <= 2 common neighbors.
        # Adding (u,v) does NOT change N(u) intersect N(v).
        # So this check is technically redundant if we maintain validity, 
        # but good for safety if starting from scratch or if logic changes.
        # However, the critical checks are the ones involving the new edge.
        
        # 2. Check constraints on neighbors of u
        # For w in N(u), u becomes a common neighbor of v and w.
        # So we need current common(v, w) <= 1 (so that adding u makes it <= 2).
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            # Calculate common neighbors of v and w
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        # 3. Check constraints on neighbors of v
        # For w in N(v), v becomes a common neighbor of u and w.
        # So we need current common(u, w) <= 1.
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_uw = adj[u] & adj[w]
            if common_uw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_fill(order):
        adj = [0] * n
        count = 0
        for u, v in order:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        return adj, count

    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    missing.append((i, j))
        return missing

    def get_present_edges(adj):
        present = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    present.append((i, j))
                mask ^= lsb
        return present

    def local_search(adj, count):
        # Simple local search: try to add any missing edge
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = get_missing_edges(adj)
            
            if not missing:
                break
                
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
                    # Continue to pack more in this pass
                    if time.time() > deadline:
                        return count
            
        return count

    def simulated_annealing(adj, count):
        """
        Simulated annealing: try random edge additions/removals with temperature-based acceptance.
        """
        if time.time() > deadline:
            return count
        
        # Tune parameters for n=20
        T = 5.0
        cooling_rate = 0.99
        min_T = 0.1
        max_steps = 2000
        
        steps = 0
        while T > min_T and steps < max_steps and time.time() < deadline - 5:
            # Pick a random move: add or remove an edge
            # Bias towards adding if we are far from max, removing to escape local optima
            if random.random() < 0.5:
                # Try to add a missing edge
                missing = get_missing_edges(adj)
                
                if not missing:
                    T *= cooling_rate
                    steps += 1
                    continue
                    
                u, v = random.choice(missing)
                
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                else:
                    # Invalid addition, skip
                    pass
            else:
                # Try to remove a present edge
                present = get_present_edges(adj)
                
                if not present:
                    T *= cooling_rate
                    steps += 1
                    continue
                    
                u, v = random.choice(present)
                remove_edge(adj, u, v)
                count -= 1
                
                # After removal, try to greedily add back as many edges as possible
                # This helps to recover lost density
                for _ in range(5):
                    if time.time() > deadline - 5:
                        break
                    missing = get_missing_edges(adj)
                    
                    if not missing:
                        break
                    
                    random.shuffle(missing)
                    added = False
                    for a, b in missing:
                        if is_valid_add(adj, a, b):
                            add_edge(adj, a, b)
                            count += 1
                            added = True
                            break
                    
                    if not added:
                        break
                
            T *= cooling_rate
            steps += 1
        
        return count

    # Run multiple random restarts
    restarts = 0
    # We have ~110 seconds. Each restart takes variable time.
    # We'll just loop until deadline.
    while time.time() < deadline - 10:
        random.shuffle(all_edges)
        adj, count = greedy_fill(all_edges)
        
        # Local search to fill in missing edges
        count = local_search(adj, count)
        
        # Simulated annealing to escape local optima
        count = simulated_annealing(adj, count)
        
        # Final polish: strict local search again
        count = local_search(adj, count)
        
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        best_edges.append([i, j])
                    mask ^= lsb
                    
        restarts += 1
        
        # If we are running out of time, break early
        if time.time() > deadline - 15:
            break

    return best_edges
