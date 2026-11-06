import time
import random
from math import gcd

def construct():
    # Use a conservative deadline to ensure we return before the 60s hard limit
    # The evaluator kills at 90s, but we want to be safe. 50s gives 10s buffer.
    start_time = time.time()
    deadline = start_time + 50
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute conflict_z[x][y] = bitmask of z such that {x,y,z} is a bad triple
    # conflict_z is a 2D array of integers (bitmasks)
    conflict_z = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Precompute list of y for each x where conflict_z[x][y] != 0
    # This speeds up the can_add check significantly
    conflict_neighbors = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            # Find all c > b such that lcm(a,c)==ab and lcm(b,c)==ab
            # Optimization: c must be a divisor of ab? No.
            # lcm(a,c) = ab => c divides ab and a*c/gcd(a,c) = ab.
            # It's faster to just iterate c from b+1 to n. N=128 is small.
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    # Triple (a, b, c) is bad
                    # Update conflict bitmasks
                    mask_c = 1 << c
                    mask_b = 1 << b
                    mask_a = 1 << a
                    
                    conflict_z[a][b] |= mask_c
                    conflict_z[b][a] |= mask_c
                    conflict_z[a][c] |= mask_b
                    conflict_z[c][a] |= mask_b
                    conflict_z[b][c] |= mask_a
                    conflict_z[c][b] |= mask_a
    
    # Build adjacency lists for fast checking
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            if x != y and conflict_z[x][y] != 0:
                conflict_neighbors[x].append(y)

    def can_add(x, S_mask):
        """Check if x can be added to set represented by S_mask."""
        # Only check y in S that have non-empty conflict with x
        for y in conflict_neighbors[x]:
            if (S_mask >> y) & 1:
                if conflict_z[x][y] & S_mask:
                    return False
        return True

    def greedy(order, S_mask=0):
        """Greedy construction."""
        S = set()
        for x in order:
            if can_add(x, S_mask):
                S.add(x)
                S_mask |= (1 << x)
        return S, S_mask

    def local_search(S, S_mask):
        """Lightweight local search: Additions and 1-for-1 swaps."""
        S = set(S)
        S_mask = S_mask
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any element
            for x in range(1, n + 1):
                if not (S_mask & (1 << x)) and can_add(x, S_mask):
                    S.add(x)
                    S_mask |= (1 << x)
                    improved = True
                    break
            
            if not improved:
                # 2. Try 1-for-1 swap
                # Remove u, add v. Then try to add more.
                # We iterate u in S, and for each u, try all v not in S.
                # This is O(|S| * N * Cost(can_add)) which is acceptable for N=128.
                s_list = list(S)
                random.shuffle(s_list)
                for u in s_list:
                    if time.time() > deadline:
                        break
                    u_bit = 1 << u
                    S_trial_mask = S_mask & ~u_bit
                    S_trial_set = S - {u}
                    
                    # Try to add v
                    for v in range(1, n + 1):
                        if not (S_trial_mask & (1 << v)) and can_add(v, S_trial_mask):
                            S_trial_set.add(v)
                            S_trial_mask |= (1 << v)
                            
                            # Check if this swap enables further additions (look-ahead 1)
                            # If yes, accept the swap and continue search
                            # If no, we might still accept it if we want to explore, 
                            # but usually we want to increase size. 
                            # Since size is same, we only accept if it leads to growth.
                            added_more = False
                            for w in range(1, n + 1):
                                if not (S_trial_mask & (1 << w)) and can_add(w, S_trial_mask):
                                    S_trial_set.add(w)
                                    S_trial_mask |= (1 << w)
                                    added_more = True
                                    break # Just one more is enough to justify
                
                            if added_more:
                                S = S_trial_set
                                S_mask = S_trial_mask
                                improved = True
                                break # Restart the while loop
                    if improved:
                        break
        
        return S, S_mask

    best_set = set()
    best_size = 0
    
    # Precompute degrees (number of conflict neighbors) for ordering
    degrees = [len(conflict_neighbors[k]) for k in range(n + 1)]
    
    # Run multiple randomized greedy passes
    # We have ~50s budget. Greedy is very fast. Local search is moderate.
    # Let's do as many as we can.
    
    while time.time() < deadline - 5:
        # Choose ordering strategy
        r = random.random()
        if r < 0.3:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif r < 0.6:
            # Least constrained first
            order = list(range(1, n + 1))
            order.sort(key=lambda x: degrees[x] + random.random() * 0.1)
        elif r < 0.8:
            # Most constrained first
            order = list(range(1, n + 1))
            order.sort(key=lambda x: -degrees[x] + random.random() * 0.1)
        else:
            # Random with bias
            order = list(range(1, n + 1))
            weights = [1.0 / (degrees[x] + 1) for x in order]
            # Simple weighted shuffle
            for pos in range(len(order) - 1, 0, -1):
                total = sum(weights[:pos + 1])
                if total == 0: continue
                rnd = random.random() * total
                cum = 0
                for k in range(pos + 1):
                    cum += weights[k]
                    if rnd <= cum:
                        order[pos], order[k] = order[k], order[pos]
                        weights[pos], weights[k] = weights[k], weights[pos]
                        break
        
        S, S_mask = greedy(order)
        
        # Apply local search
        S, S_mask = local_search(S, S_mask)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Final polish on the best set found
    if best_set:
        best_set, _ = local_search(best_set, 0)
        # Reconstruct mask
        m = 0
        for x in best_set:
            m |= (1 << x)
        best_set, _ = local_search(best_set, m)
    
    return list(best_set)
