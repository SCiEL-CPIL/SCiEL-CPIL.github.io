import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 55.0

    # Precompute conflict bitmasks
    # conflict_mask[x] has bits set for all elements that conflict with x
    # i.e., if x and y are both in the set, they form a 4-AP if y is in conflict_mask[x]
    # Actually, we need: can_add(mask, x) checks if mask & conflict_mask[x] == 0
    # But a 4-AP requires 3 other elements. So conflict is not pairwise.
    # We need to check if there exist a,b,c in mask such that {a,b,c,x} is a 4-AP.
    
    # Let's stick with the triple approach but optimize it.
    # triples[x] = list of (a, b, c) such that {a,b,c,x} is a 4-AP
    
    # Alternative: Precompute for each x, a list of pairs (a,b) such that 
    # if a and b are in the set, then x cannot be added.
    # This reduces the check from O(3) bitwise ANDs to O(2) or even simpler.
    # Actually, if we have pairs (a,b), we check if (mask & (1<<a)) and (mask & (1<<b)).
    # This is still similar cost.
    
    # Let's try a different optimization: 
    # For each x, precompute a bitmask `conflict_pairs_mask` is not directly possible 
    # because it's a 3-SAT like condition (x conflicts if ANY triple is satisfied).
    
    # However, we can precompute for each x, the list of triples as a list of 64-bit ints 
    # representing the mask of the 3 elements.
    # triple_mask = (1 << a) | (1 << b) | (1 << c)
    # Then can_add is: for m in triple_masks[x]: if mask & m == m: return False
    
    # This might be faster than 3 separate shifts and ANDs.
    
    triple_masks = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        # Case 1: x is 1st term (x, x+d, x+2d, x+3d)
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            a, b, c = x + d, x + 2 * d, x + 3 * d
            triple_masks[x].append((1 << a) | (1 << b) | (1 << c))
        
        # Case 2: x is 2nd term (x-d, x, x+d, x+2d)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            a, b, c = x - d, x + d, x + 2 * d
            triple_masks[x].append((1 << a) | (1 << b) | (1 << c))
            
        # Case 3: x is 3rd term (x-2d, x-d, x, x+d)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            a, b, c = x - 2 * d, x - d, x + d
            triple_masks[x].append((1 << a) | (1 << b) | (1 << c))
            
        # Case 4: x is 4th term (x-3d, x-2d, x-d, x)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            a, b, c = x - 3 * d, x - 2 * d, x - d
            triple_masks[x].append((1 << a) | (1 << b) | (1 << c))

    def can_add(mask, x):
        for m in triple_masks[x]:
            if (mask & m) == m:
                return False
        return True

    def randomized_greedy(seed):
        """Generate a set using randomized greedy with shuffled order."""
        rng = random.Random(seed)
        mask = 0
        order = list(range(1, n + 1))
        rng.shuffle(order)
        for x in order:
            if can_add(mask, x):
                mask |= (1 << x)
        return mask

    def greedy_fill(mask, order):
        for x in order:
            if not ((mask >> x) & 1):
                if can_add(mask, x):
                    mask |= (1 << x)
        return mask

    # Phase 1: Generate diverse initial seeds using randomized greedy
    # This allows us to explore different regions of the search space
    best_mask = 0
    best_size = 0
    
    # First, try the base-3 Salem-Spencer set as a strong baseline
    init_mask = 0
    powers = [1, 3, 9, 27, 81, 243]
    for i in range(1, 1 << len(powers)):
        val = 0
        for j, p in enumerate(powers):
            if (i >> j) & 1:
                val += p
        if 1 <= val <= n:
            init_mask |= (1 << val)
    
    best_mask = init_mask
    best_size = bin(init_mask).count('1')
    
    # Generate multiple randomized greedy seeds
    num_seeds = 50
    for seed in range(num_seeds):
        if time.time() - start_time > time_limit:
            break
        mask = randomized_greedy(seed)
        size = bin(mask).count('1')
        if size > best_size:
            best_mask = mask
            best_size = size
    
    random.seed(42)
    
    # Local Search starting from best seed
    current_mask = best_mask
    current_size = best_size
    
    # Precompute list of all elements 1..n
    all_elements = list(range(1, n + 1))
    
    while time.time() - start_time < time_limit:
        # 1. Try to add any element
        added = False
        # Randomize candidates to avoid bias
        candidates = [i for i in all_elements if not ((current_mask >> i) & 1)]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(current_mask, x):
                current_mask |= (1 << x)
                current_size += 1
                added = True
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
                break
        
        if not added:
            # 2. Kick: Remove k random elements
            # Choose k based on current size or random
            # Increase kick strength occasionally
            if random.random() < 0.1:
                k = random.randint(5, 10)
            else:
                k = random.randint(1, 5)
            
            if current_size > k:
                # Get list of elements in current set
                elements_in_set = [i for i in all_elements if (current_mask >> i) & 1]
                to_remove = random.sample(elements_in_set, min(k, current_size))
                for x in to_remove:
                    current_mask &= ~(1 << x)
                    current_size -= 1
                
                # 3. Greedy Refill
                # Try to add all possible elements in random order
                candidates = [i for i in all_elements if not ((current_mask >> i) & 1)]
                random.shuffle(candidates)
                for x in candidates:
                    if can_add(current_mask, x):
                        current_mask |= (1 << x)
                        current_size += 1
                
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            
            # 4. Occasionally restart from a new randomized greedy seed
            if random.random() < 0.05 and time.time() - start_time < time_limit - 5:
                new_seed = random.randint(1000, 100000)
                new_mask = randomized_greedy(new_seed)
                new_size = bin(new_mask).count('1')
                if new_size > current_size:
                    current_mask = new_mask
                    current_size = new_size

    # Convert mask to sorted list
    result = [i for i in range(1, n + 1) if (best_mask >> i) & 1]
    return result
