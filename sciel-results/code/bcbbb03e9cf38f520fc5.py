import time
import random
import itertools
import numpy as np

def _paley_17():
    """Return the Paley graph on 17 vertices as a bitmask adjacency list."""
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _has_k4(adj):
    n = len(adj)
    for i in range(n - 3):
        # Neighbors of i greater than i
        ni = adj[i] & ~((1 << (i + 1)) - 1)
        mask_j = ni
        while mask_j:
            j_bit = mask_j & -mask_j
            j = j_bit.bit_length() - 1
            mask_j ^= j_bit
            
            # Neighbors of j greater than j, also neighbors of i
            nij = adj[j] & adj[i] & ~((1 << (j + 1)) - 1)
            mask_k = nij
            while mask_k:
                k_bit = mask_k & -mask_k
                k = k_bit.bit_length() - 1
                mask_k ^= k_bit
                
                # Check if i, j, k have a common neighbor l > k
                # i, j, k are mutually connected (triangle)
                # We need l connected to i, j, k
                common = adj[i] & adj[j] & adj[k] & ~((1 << (k + 1)) - 1)
                if common:
                    return True
    return False

def _has_ind6(adj):
    n = len(adj)
    # Check for independent set of size 6
    # Equivalent to K6 in complement graph
    # Iterate 5-combinations. If a 5-set is independent, check if it has a common non-neighbor
    for combo in itertools.combinations(range(n), 5):
        # Check if combo is independent
        is_ind = True
        mask5 = 0
        for v in combo:
            mask5 |= (1 << v)
            # Optimization: check against previous vertices in combo
            # Actually, just check all pairs?
            # For speed, check if adj[v] intersects with mask of previous vertices
            pass
        
        # Proper check for independence of combo
        # Since n is small, this is fine
        for idx, v in enumerate(combo):
            # Check against other vertices in combo
            others_mask = 0
            for u in combo:
                if u != v:
                    others_mask |= (1 << u)
            if adj[v] & others_mask:
                is_ind = False
                break
        
        if is_ind:
            # Find common non-neighbor of the 5 vertices
            # A vertex x is a non-neighbor of all v in combo if (adj[v] & (1<<x)) == 0 for all v
            # This is equivalent to: x is NOT in the union of neighbors of combo
            # Union of neighbors:
            union_neighbors = 0
            for v in combo:
                union_neighbors |= adj[v]
            
            # Candidates for x are vertices not in combo and not in union_neighbors
            # Also x > max(combo) to avoid duplicates? 
            # We just need to find ANY x.
            # To avoid finding the same set multiple times, we can restrict x > max(combo)
            # But if we find ANY x, we return True.
            # So we can just check if there is ANY vertex not in combo and not in union_neighbors.
            all_vertices_mask = (1 << n) - 1
            candidates = all_vertices_mask & ~mask5 & ~union_neighbors
            
            if candidates:
                return True
    return False

def _is_valid(adj):
    return not _has_k4(adj) and not _has_ind6(adj)

def _sa_find(n, time_budget=5.0):
    """Run Simulated Annealing to find a valid graph on n vertices."""
    start_time = time.time()
    best_cost = float('inf')
    best_adj = None
    
    # Initial random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Calculate initial cost (number of violations)
    # We use a simple cost: 10 if K4 exists, 10 if Ind6 exists?
    # Better: count violations?
    # For speed, let's just use boolean validity for acceptance, 
    # but we need a cost gradient for SA.
    # Let's define cost as:
    # If has_k4: cost += 100
    # If has_ind6: cost += 100
    # This is coarse.
    # Alternative: Just use "valid" or "invalid".
    # SA with binary cost is like random walk with restarts.
    # Let's try to count K4s and Ind6s but fast.
    
    def get_cost(adj):
        c = 0
        if _has_k4(adj):
            c += 100
        if _has_ind6(adj):
            c += 100
        return c

    current_adj = adj[:]
    current_cost = get_cost(current_adj)
    if current_cost == 0:
        return current_adj
    
    T = 1.0
    cooling = 0.995
    best_adj = current_adj[:]
    best_cost = current_cost
    
    while time.time() - start_time < time_budget:
        if n < 2:
            break
        # Pick a random edge to flip
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        
        # Flip edge
        if (current_adj[i] >> j) & 1:
            # Remove edge
            current_adj[i] &= ~(1 << j)
            current_adj[j] &= ~(1 << i)
        else:
            # Add edge
            current_adj[i] |= (1 << j)
            current_adj[j] |= (1 << i)
        
        new_cost = get_cost(current_adj)
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = current_adj[:]
                if best_cost == 0:
                    return best_adj
        else:
            # Revert
            if (current_adj[i] >> j) & 1:
                # It was removed, so add back
                current_adj[i] |= (1 << j)
                current_adj[j] |= (1 << i)
            else:
                # It was added, so remove
                current_adj[i] &= ~(1 << j)
                current_adj[j] &= ~(1 << i)
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_adj if best_cost == 0 else None

def construct():
    deadline = time.time() + 110
    best_n = 17
    best_adj = _paley_17()
    
    # Try to improve n
    # We know 17 is valid.
    # Try 18, 19, ...
    for n in range(18, 36):
        if time.time() > deadline - 15:
            break
        
        found = None
        # Use more restarts for smaller n, fewer for larger?
        # Or just fixed time budget per n.
        # Let's allocate time based on remaining budget.
        remaining = deadline - time.time() - 10
        if remaining < 5:
            break
            
        # Try a few restarts
        # For n=18, it should be easy. For n=30, hard.
        # Let's try 5 restarts of 2s each?
        # Or 1 long run?
        # SA benefits from restarts.
        num_restarts = 3
        time_per_restart = min(3.0, remaining / num_restarts)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            adj = _sa_find(n, time_budget=time_per_restart)
            if adj:
                found = adj
                break
        
        if found:
            best_n = n
            best_adj = found
        else:
            # If we fail at n, it's likely to fail at n+1?
            # Not necessarily, but let's continue.
            pass

    # Convert bitmask to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
