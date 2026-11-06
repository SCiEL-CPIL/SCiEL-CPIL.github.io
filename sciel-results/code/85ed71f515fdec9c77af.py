import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    # Precompute conflict bitmasks
    # conflicts[x] is a list of bitmasks (each with 3 bits set) representing
    # the other 3 elements required to form a 4-AP with x.
    # If any of these masks is a subset of the current set mask, x cannot be added.
    conflicts = [[] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        # Case 1: x is 1st term (x, x+d, x+2d, x+3d)
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            a, b, c = x + d, x + 2 * d, x + 3 * d
            if a <= n and b <= n and c <= n:
                conflicts[x].append((1 << a) | (1 << b) | (1 << c))
        
        # Case 2: x is 2nd term (x-d, x, x+d, x+2d)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            a, b, c = x - d, x + d, x + 2 * d
            if a >= 1 and b <= n and c <= n:
                conflicts[x].append((1 << a) | (1 << b) | (1 << c))
            
        # Case 3: x is 3rd term (x-2d, x-d, x, x+d)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            a, b, c = x - 2 * d, x - d, x + d
            if a >= 1 and b >= 1 and c <= n:
                conflicts[x].append((1 << a) | (1 << b) | (1 << c))
            
        # Case 4: x is 4th term (x-3d, x-2d, x-d, x)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            a, b, c = x - 3 * d, x - 2 * d, x - d
            if a >= 1 and b >= 1 and c >= 1:
                conflicts[x].append((1 << a) | (1 << b) | (1 << c))

    def can_add(mask, x):
        # Check if x can be added to mask without forming a 4-AP
        for m in conflicts[x]:
            if (mask & m) == m:
                return False
        return True

    def greedy_fill(mask, rng):
        """Greedily add elements in random order until no more can be added."""
        all_elements = list(range(1, n + 1))
        rng.shuffle(all_elements)
        for x in all_elements:
            if not ((mask >> x) & 1):
                if can_add(mask, x):
                    mask |= (1 << x)
        return mask

    # Initial construction using base-3 (Salem-Spencer)
    # This provides a strong 4-AP free set to start with
    init_mask = 0
    # Elements with only 0s and 1s in base 3 representation are 3-AP free
    # 3^0=1, 3^1=3, 3^2=9, 3^3=27, 3^4=81, 3^5=243
    # Max sum is 1+3+9+27+81+243 = 364 <= 512
    # We can also include some 2s? No, 3-AP free sets usually use digits 0,1.
    # Let's stick to 0,1 digits for safety.
    powers = [1, 3, 9, 27, 81, 243]
    for i in range(1, 1 << len(powers)):
        val = 0
        for j, p in enumerate(powers):
            if (i >> j) & 1:
                val += p
        if 1 <= val <= n:
            init_mask |= (1 << val)
    
    current_mask = init_mask
    current_size = bin(current_mask).count('1')
    best_mask = current_mask
    best_size = current_size

    rng = random.Random(42)
    all_elements = list(range(1, n + 1))
    
    # Local Search Loop
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Try to add any element
        candidates = [i for i in all_elements if not ((current_mask >> i) & 1)]
        rng.shuffle(candidates)
        for x in candidates:
            if can_add(current_mask, x):
                current_mask |= (1 << x)
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
                break
        
        if not improved:
            # Phase 2: Try Swap Moves
            # Pick a random element to remove, then try to add a new one
            elements_in_set = [i for i in all_elements if (current_mask >> i) & 1]
            if elements_in_set:
                swap_found = False
                # Try a few random swaps
                for _ in range(20):  # Limit swap attempts to avoid wasting time
                    if time.time() - start_time > time_limit:
                        break
                    u = rng.choice(elements_in_set)
                    temp_mask = current_mask & ~(1 << u)
                    
                    # Try to add any element to temp_mask
                    # To be efficient, we can just try a few random candidates
                    # or iterate through all non-set elements?
                    # Iterating all is O(N). N=512. 512 * 20 swaps = 10k checks.
                    # Each check is ~300 ops. 3M ops. ~0.1s. Acceptable.
                    
                    candidates_for_swap = [i for i in all_elements if not ((temp_mask >> i) & 1) and i != u]
                    rng.shuffle(candidates_for_swap)
                    
                    added_v = None
                    for v in candidates_for_swap:
                        if can_add(temp_mask, v):
                            added_v = v
                            break
                    
                    if added_v is not None:
                        # Perform the swap
                        current_mask = temp_mask | (1 << added_v)
                        # Size remains the same, but we might be able to add more now
                        # So we loop back to Phase 1
                        swap_found = True
                        break
                
                if not swap_found:
                    # Phase 3: Kick (Perturb)
                    # Remove k random elements and refill
                    k = rng.randint(3, 8)
                    if current_size > k:
                        to_remove = rng.sample(elements_in_set, k)
                        for x in to_remove:
                            current_mask &= ~(1 << x)
                        current_size -= k
                        
                        # Greedy Refill
                        current_mask = greedy_fill(current_mask, rng)
                        current_size = bin(current_mask).count('1')
                        
                        if current_size > best_size:
                            best_mask = current_mask
                            best_size = current_size
                    else:
                        # If set is small, restart with a new random greedy seed
                        new_mask = 0
                        order = list(range(1, n + 1))
                        rng.shuffle(order)
                        for x in order:
                            if can_add(new_mask, x):
                                new_mask |= (1 << x)
                        new_size = bin(new_mask).count('1')
                        if new_size > current_size:
                            current_mask = new_mask
                            current_size = new_size

    # Convert mask to sorted list
    result = [i for i in range(1, n + 1) if (best_mask >> i) & 1]
    return result
