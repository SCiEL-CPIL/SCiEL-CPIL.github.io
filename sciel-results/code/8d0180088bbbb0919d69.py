import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 256
    
    # Precompute prime factor sets for each number
    def prime_factors(x):
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors
    
    # Get prime factors for all numbers 1..256
    pfs = [set()]  # index 0 unused
    for i in range(1, n + 1):
        pfs.append(prime_factors(i))
    
    # Build conflict graph: two numbers conflict if they share a prime factor
    # We'll use bitmasks for the conflict sets
    # Number of numbers is 256, so we need 256-bit integers (Python handles big ints natively)
    
    # conflict_mask[i] is a bitmask where bit j is set if number (i+1) conflicts with number (j+1)
    # Actually, let's use 0-indexed internally: index i represents number i+1
    # Let's define: for number i (1-indexed), its index in our array is i-1
    
    # Build adjacency: adj[i] is a bitmask of which other numbers conflict with number at position i (0-indexed, representing value i+1)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if pfs[i + 1] & pfs[j + 1]:  # share a prime factor
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Values: value[i] = i + 1
    values = [i + 1 for i in range(n)]
    
    # Branch and bound for maximum weight independent set
    # We process vertices in order of decreasing value (to find good solutions early)
    # Order: sort indices by value descending
    order = sorted(range(n), key=lambda i: values[i], reverse=True)
    
    # Re-index: new_index[k] = old_index of the k-th vertex in our processing order
    # pos_in_order[old_idx] = position in the order
    new_index = order
    pos_in_order = [0] * n
    for k, old_idx in enumerate(order):
        pos_in_order[old_idx] = k
    
    # Rebuild adjacency in the new order
    # new_adj[k] is a bitmask of positions > k that conflict with position k
    # (we only need forward edges for the recursion)
    new_adj = [0] * n
    for k in range(n):
        old_idx = new_index[k]
        mask = 0
        for l in range(k + 1, n):
            old_idx2 = new_index[l]
            if adj[old_idx] & (1 << old_idx2):
                mask |= (1 << (l - k - 1))  # relative position
        new_adj[k] = mask
    
    # Values in the new order
    ordered_values = [values[new_index[k]] for k in range(n)]
    
    # For upper bound: sum of remaining values
    suffix_sum = [0] * (n + 1)
    for k in range(n - 1, -1, -1):
        suffix_sum[k] = suffix_sum[k + 1] + ordered_values[k]
    
    best_sum = 0
    best_set = [False] * n  # in new order
    
    def branch_and_bound(k, current_sum, selected_mask):
        """
        k: current position in the order
        current_sum: sum of selected values so far
        selected_mask: bitmask of which positions >= k have been selected
        """
        nonlocal best_sum, best_set
        
        # Deadline check
        if time.time() > deadline:
            return
        
        # Upper bound: current_sum + sum of all remaining values
        if current_sum + suffix_sum[k] <= best_sum:
            return
        
        # If we've processed all vertices
        if k == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = [bool(selected_mask & (1 << i)) for i in range(n)]
            return
        
        # Option 1: Skip vertex k
        branch_and_bound(k + 1, current_sum, selected_mask)
        
        # Option 2: Include vertex k if it doesn't conflict with any already selected
        # Check if vertex k conflicts with any selected vertex
        # A vertex at position l (l > k) conflicts with k if bit (l - k - 1) is set in new_adj[k]
        # And l is selected if bit l is set in selected_mask
        # We need to check: for all l where new_adj[k] has bit (l-k-1), selected_mask doesn't have bit l
        
        # Actually, let's think about this differently.
        # When we're at position k, we've made decisions for positions 0..k-1.
        # selected_mask represents decisions for positions >= k (not yet made).
        # Wait, I need to reconsider the recursion structure.
        
        # Let me restructure: 
        # At position k, we decide whether to include vertex k.
        # If we include it, we must exclude all vertices that conflict with k.
        # The conflict mask tells us which future vertices conflict with k.
        
        # For including k: check that no already-included vertex conflicts with k.
        # But in this formulation, we process in order, so when we're at k, 
        # all previously processed vertices (0..k-1) have been decided.
        # We need to track which of those are selected and check conflicts.
        
        # This is getting complex. Let me use a simpler approach:
        # Track the set of selected vertices as a bitmask over the original indices.
        # But that's 256 bits, which is fine in Python.
        
        pass
    
    # Let me use a cleaner recursive approach with a set of available vertices
    # Actually, let's use the standard maximum independent set branch and bound
    
    # Simpler approach: use a greedy initialization, then local search with multiple restarts
    
    # Greedy: pick highest value vertex that doesn't conflict with already picked
    def greedy_solve():
        available = set(range(n))
        selected = []
        total = 0
        while available:
            # Pick the vertex with highest value
            best = max(available, key=lambda i: ordered_values[i])
            selected.append(best)
            total += ordered_values[best]
            # Remove best and all its conflicts
            to_remove = {best}
            adj_mask = new_adj[best]
            for l in range(best + 1, n):
                if adj_mask & (1 << (l - best - 1)):
                    to_remove.add(l)
            available -= to_remove
        return selected, total
    
    # Initial greedy solution
    greedy_selected, greedy_sum = greedy_solve()
    best_sum = greedy_sum
    best_positions = set(greedy_selected)
    
    # Local search: try swapping one selected vertex for an unselected one, 
    # or adding/removing vertices
    def local_search(selected_set, current_sum, max_iterations=50000):
        nonlocal best_sum, best_positions
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            iteration += 1
            
            # Try adding a vertex
            for v in range(n):
                if v in selected_set:
                    continue
                # Check if v conflicts with any selected vertex
                conflict = False
                for u in selected_set:
                    if u < v:
                        if new_adj[u] & (1 << (v - u - 1)):
                            conflict = True
                            break
                    elif u > v:
                        if new_adj[v] & (1 << (u - v - 1)):
                            conflict = True
                            break
                if not conflict:
                    new_set = selected_set | {v}
                    new_sum = current_sum + ordered_values[v]
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_positions = new_set
                    selected_set = new_set
                    current_sum = new_sum
                    improved = True
                    break
            
            # Try swapping: remove one selected, add one unselected
            if not improved:
                for u in list(selected_set):
                    for v in range(n):
                        if v in selected_set:
                            continue
                        # Check if v conflicts with any selected except u
                        conflict = False
                        for w in selected_set:
                            if w == u:
                                continue
                            if w < v:
                                if new_adj[w] & (1 << (v - w - 1)):
                                    conflict = True
                                    break
                            elif w > v:
                                if new_adj[v] & (1 << (w - v - 1)):
                                    conflict = True
                                    break
                        if not conflict:
                            new_set = (selected_set - {u}) | {v}
                            new_sum = current_sum - ordered_values[u] + ordered_values[v]
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_positions = new_set
                            selected_set = new_set
                            current_sum = new_sum
                            improved = True
                            break
                    if improved:
                        break
        
        return selected_set, current_sum
    
    # Run local search from greedy solution
    local_search(set(greedy_selected), greedy_sum)
    
    # Try multiple greedy orders (randomized)
    import random
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Randomized greedy: shuffle vertices with small values to get different solutions
        shuffled_order = list(range(n))
        random.shuffle(shuffled_order)
        # But keep high-value vertices in their natural order mostly
        # Actually, let's just do a simple random restart with different tie-breaking
        
        available = set(range(n))
        selected = []
        total = 0
        while available:
            # Pick a random vertex from the top 10% by value
            sorted_avail = sorted(available, key=lambda i: ordered_values[i], reverse=True)
            top_k = max(1, len(sorted_avail) // 10)
            candidates = sorted_avail[:top_k]
            best = random.choice(candidates)
            selected.append(best)
            total += ordered_values[best]
            to_remove = {best}
            adj_mask = new_adj[best]
            for l in range(best + 1, n):
                if adj_mask & (1 << (l - best - 1)):
                    to_remove.add(l)
            available -= to_remove
        
        # Local search from this solution
        local_search(set(selected), total)
    
    # Convert best_positions back to original values
    result = [ordered_values[k] for k in sorted(best_positions)]
    
    # Verify the solution is valid
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Fallback to prime powers
                pp_solution = [1]
                for p in range(2, n + 1):
                    if all(p % q != 0 for q in range(2, int(math.sqrt(p)) + 1)):
                        power = p
                        while power * p <= n:
                            power *= p
                        pp_solution.append(power)
                return [int(x) for x in pp_solution]
    
    return [int(x) for x in result]
