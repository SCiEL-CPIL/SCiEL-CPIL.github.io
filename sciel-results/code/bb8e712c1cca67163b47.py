import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for the 90s limit
    
    n = 128
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    # 1. Precompute conflict structures
    # bad_triples: list of (a, b, c) such that lcm(a,b)==lcm(a,c)==lcm(b,c)
    # conflict_pairs[x]: set of (u, v) such that {u, v, x} is a bad triple.
    # This means if u and v are in the set, x cannot be added.
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == L_ab and _lcm(b, c) == L_ab:
                    bad_triples.append((a, b, c))
                    
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
        
    # Precompute degrees (number of conflicting pairs) for each element
    degrees = [len(conflict_pairs[i]) for i in range(n + 1)]
    
    # Bitmask representation: element i is represented by bit (i-1)
    # mask is an integer where bit j is 1 if element j+1 is in the set.
    
    def can_add(x, mask):
        """Check if element x can be added to the set represented by mask."""
        for u, v in conflict_pairs[x]:
            # Check if both u and v are in mask
            # Bit index for element k is k-1
            if (mask >> (u - 1)) & 1 and (mask >> (v - 1)) & 1:
                return False
        return True

    def popcount(mask):
        return bin(mask).count('1')
        
    def greedy(order, mask=0):
        """Greedy construction given an order of elements."""
        selected = mask
        for x in order:
            if not (selected >> (x - 1)) & 1:
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected

    def local_search(mask):
        """
        Local search to maximize the size of the independent set.
        Moves:
        1. Add any valid element (size +1).
        2. Remove 1, Add 2 (size +1).
        3. Remove 2, Add 3 (size +1).
        """
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add any missing element
            for x in range(1, n + 1):
                if not (mask >> (x - 1)) & 1:
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
                        break
            if improved:
                continue

            # Get list of selected elements
            s_list = [i for i in range(1, n + 1) if (mask >> (i - 1)) & 1]
            
            # 2. Try 1-for-2 swap: remove x, add y, z
            # We iterate over x in s_list. For each x, we remove it, 
            # then try to add any two elements from the complement.
            # To be efficient, we can just try to add elements greedily 
            # from the complement after removing x. If we can add 2, we win.
            
            found_swap = False
            for x in s_list:
                if time.time() > deadline - 2:
                    break
                mask_no_x = mask & ~(1 << (x - 1))
                # Try to add elements
                added_count = 0
                new_mask = mask_no_x
                # We want to find 2 elements to add.
                # Iterate through all elements not in mask_no_x
                # Sort by degree? Or random? 
                # Since we just want ANY 2, we can just iterate 1..n
                # But we must ensure the 2 added are valid with each other.
                # can_add(y, new_mask) ensures y is valid with current new_mask.
                # So sequential addition works.
                
                # Optimization: Pre-sort candidates by degree to find "easy" adds?
                # For now, simple iteration.
                for y in range(1, n + 1):
                    if y == x:
                        continue
                    if not (new_mask >> (y - 1)) & 1:
                        if can_add(y, new_mask):
                            new_mask |= (1 << (y - 1))
                            added_count += 1
                            if added_count == 2:
                                break
                            
                if added_count >= 2:
                    # Size increased by 1
                    mask = new_mask
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                continue

            # 3. Try 2-for-3 swap: remove x, y, add z, w, v
            # This is O(N^3) effectively if done naively.
            # We can limit the scope: only try if we have time.
            # Or use a heuristic: pick a random pair to remove.
            
            if len(s_list) >= 2:
                # Try a few random 2-for-3 swaps
                # Number of trials based on remaining time?
                # Let's try up to 50 random pairs of removals
                for _ in range(50):
                    if time.time() > deadline - 2:
                        break
                    # Pick two distinct elements to remove
                    i, j = random.sample(range(len(s_list)), 2)
                    x = s_list[i]
                    y = s_list[j]
                    if x == y: continue
                    
                    mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                    added_count = 0
                    new_mask = mask_no_xy
                    
                    # Try to add 3 elements
                    # We need to be careful: the 3 elements must be distinct from each other and from x,y
                    # can_add handles validity with current mask.
                    
                    # To find 3, we might need to be smarter.
                    # Simple greedy: iterate 1..n
                    for z in range(1, n + 1):
                        if z == x or z == y:
                            continue
                        if not (new_mask >> (z - 1)) & 1:
                            if can_add(z, new_mask):
                                new_mask |= (1 << (z - 1))
                                added_count += 1
                                if added_count == 3:
                                    break
                                    
                    if added_count >= 3:
                        mask = new_mask
                        improved = True
                        break
                        
            if not improved:
                # Stuck in local optimum. 
                # Apply random perturbation to escape?
                # Or just break and let the main loop restart.
                # Let's do a random perturbation: remove a random element, add a random valid one?
                # Or just break. The main loop will do a new greedy restart.
                break
                
        return mask

    def simulated_annealing(mask, steps=1000):
        """
        Simulated annealing to escape local optima.
        Moves: Flip a bit (add or remove).
        Energy: -size (we want to minimize energy, i.e., maximize size).
        But we must maintain validity.
        So:
        - If adding is valid, accept always.
        - If removing, accept with probability exp(-delta_E / T).
          delta_E = -1 (size decreases). So exp(1/T).
          Actually, standard SA: accept if new_energy < old_energy OR with prob exp((new-old)/T).
          Here new_size = old_size - 1. delta_size = -1.
          We want to maximize size.
          Let's define Cost = -Size.
          If we remove, Cost increases by 1.
          Accept with prob exp(-1/T).
        """
        T = 1.0
        cooling = 0.99
        current_size = popcount(mask)
        
        for _ in range(steps):
            if time.time() > deadline - 2:
                break
                
            # Pick a random element
            x = random.randint(1, n)
            bit = 1 << (x - 1)
            
            if mask & bit:
                # Remove x
                new_mask = mask & ~bit
                # Always valid to remove
                new_size = current_size - 1
                delta_cost = 1 # Cost increases by 1
                if random.random() < pow(2.718281828, -delta_cost / T):
                    mask = new_mask
                    current_size = new_size
            else:
                # Add x
                if can_add(x, mask):
                    mask |= bit
                    current_size += 1
                # Else reject (invalid)
            
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        return mask

    best_mask = 0
    best_size = 0
    
    # Main loop: Multi-start Greedy + Local Search + SA
    # We have ~85 seconds.
    # Each greedy is O(N^2) ~ 16000 ops. Very fast.
    # Local search is the bottleneck.
    
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        
        # Generate order
        # Strategy 1: Random
        # Strategy 2: Sorted by degree (ascending)
        # Strategy 3: Sorted by degree (descending)
        # Strategy 4: Random with bias
        
        r = random.random()
        if r < 0.3:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif r < 0.6:
            order = sorted(range(1, n + 1), key=lambda x: degrees[x])
        elif r < 0.8:
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x])
        else:
            # Weighted random
            weights = [1.0 / (degrees[i] + 1) for i in range(1, n + 1)]
            order = [0] * n
            pool = list(range(1, n + 1))
            for i in range(n):
                # Select based on weights
                total_w = sum(weights[j-1] for j in pool)
                r_val = random.random() * total_w
                cum = 0
                for j, elem in enumerate(pool):
                    cum += weights[elem-1]
                    if r_val <= cum:
                        order[i] = elem
                        pool.pop(j)
                        break
                        
        mask = greedy(order)
        
        # Local Search
        mask = local_search(mask)
        
        # Simulated Annealing to escape local optima
        # Only if we have time
        if time.time() < deadline - 10:
            # Run SA for a short duration
            sa_steps = 500
            mask = simulated_annealing(mask, steps=sa_steps)
            # Polish with local search again
            mask = local_search(mask)
            
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Final polish
    best_mask = local_search(best_mask)
    
    result = [i for i in range(1, n + 1) if (best_mask >> (i - 1)) & 1]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

# Helper for popcount if not using bin (bin is fast enough for 128 bits)
# Already defined inside construct

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Result size: {len(res)}")
    print(res)
