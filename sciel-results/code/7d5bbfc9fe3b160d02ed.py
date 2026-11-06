import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_add(adj, u, v):
        # adj is a list of integers (bitmasks)
        # 1. Check common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # 2. Check constraints on neighbors of u
        # For w in N(u), u becomes a common neighbor of v and w.
        # So we need current common(v, w) <= 1.
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

    def local_search(adj, count):
        # Simple local search: try to add edges that were skipped or swap
        # For now, just try to add any missing edge
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find all missing edges
            missing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1:
                        missing.append((i, j))
            
            if not missing:
                break
                
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
                    # Continue to pack more
                    if time.time() > deadline:
                        return count
            
        return count

    def simulated_annealing(adj, count):
        """
        Simulated annealing: try random edge additions/removals with temperature-based acceptance.
        """
        if time.time() > deadline:
            return count
        
        T = 2.0
        cooling_rate = 0.999
        min_T = 0.01
        
        while T > min_T and time.time() < deadline - 2:
            # Pick a random move: add or remove an edge
            if random.random() < 0.6:
                # Try to add a missing edge
                missing = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj[i] >> j) & 1:
                            missing.append((i, j))
                
                if not missing:
                    T *= cooling_rate
                    continue
                    
                u, v = random.choice(missing)
                
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    T *= cooling_rate
                else:
                    # Invalid addition, skip
                    T *= cooling_rate
            else:
                # Try to remove a present edge (to escape local optima)
                present = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            present.append((i, j))
                        mask ^= lsb
                
                if not present:
                    T *= cooling_rate
                    continue
                    
                u, v = random.choice(present)
                remove_edge(adj, u, v)
                count -= 1
                
                # After removal, try to greedily add back as many edges as possible
                for _ in range(3):
                    if time.time() > deadline - 2:
                        break
                    missing = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if not (adj[i] >> j) & 1:
                                missing.append((i, j))
                    
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
        
        return count

    # Run multiple random restarts with simulated annealing
    restarts = 0
    while time.time() < deadline - 10:
        random.shuffle(all_edges)
        adj, count = greedy_fill(all_edges)
        
        # Local search to fill in missing edges
        count = local_search(adj, count)
        
        # Simulated annealing to escape local optima
        count = simulated_annealing(adj, count)
        
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
