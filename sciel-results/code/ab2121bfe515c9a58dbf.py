import random
import time

def construct():
    n = 512
    
    # Precompute valid d values for each x and each position (1st, 2nd, 3rd, 4th)
    # d_1st[x] = list of d such that x, x+d, x+2d, x+3d are in [1, n]
    # d_2nd[x] = list of d such that x-d, x, x+d, x+2d are in [1, n]
    # d_3rd[x] = list of d such that x-2d, x-d, x, x+d are in [1, n]
    # d_4th[x] = list of d such that x-3d, x-2d, x-d, x are in [1, n]
    
    d1 = [[] for _ in range(n+1)]
    d2 = [[] for _ in range(n+1)]
    d3 = [[] for _ in range(n+1)]
    d4 = [[] for _ in range(n+1)]
    
    for x in range(1, n+1):
        # Case 1: x is 1st. x+3d <= n => d <= (n-x)//3
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            d1[x].append(d)
            
        # Case 2: x is 2nd. x-d >= 1 and x+2d <= n => d <= min(x-1, (n-x)//2)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            d2[x].append(d)
            
        # Case 3: x is 3rd. x-2d >= 1 and x+d <= n => d <= min((x-1)//2, n-x)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            d3[x].append(d)
            
        # Case 4: x is 4th. x-3d >= 1 => d <= (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            d4[x].append(d)

    def can_add(x, m):
        # Check if adding x creates a 4-AP
        # 1. x is 1st
        for d in d1[x]:
            if ((m >> (x + d)) & 1) and ((m >> (x + 2*d)) & 1) and ((m >> (x + 3*d)) & 1):
                return False
        # 2. x is 2nd
        for d in d2[x]:
            if ((m >> (x - d)) & 1) and ((m >> (x + d)) & 1) and ((m >> (x + 2*d)) & 1):
                return False
        # 3. x is 3rd
        for d in d3[x]:
            if ((m >> (x - 2*d)) & 1) and ((m >> (x - d)) & 1) and ((m >> (x + d)) & 1):
                return False
        # 4. x is 4th
        for d in d4[x]:
            if ((m >> (x - 3*d)) & 1) and ((m >> (x - 2*d)) & 1) and ((m >> (x - d)) & 1):
                return False
        return True

    def get_size(m):
        return bin(m).count('1')

    # Step 1: Base-3 digit-based construction (Cantor Set)
    mask = 0
    for x in range(1, n + 1):
        num = x
        valid = True
        while num > 0:
            if num % 3 == 2:
                valid = False
                break
            num //= 3
        if valid:
            mask |= (1 << x)

    def greedy_add(m, order):
        for x in order:
            if not ((m >> x) & 1):
                if can_add(x, m):
                    m |= (1 << x)
        return m

    # Initial Greedy Pass with standard order
    mask = greedy_add(mask, list(range(1, n + 1)))
    best_mask = mask
    best_size = get_size(mask)
    
    start_time = time.time()
    time_limit = 55.0
    
    # Local Search: Randomized Hill Climbing
    # We will try to improve the set by removing one element and greedily refilling.
    # To escape local minima, we also try removing 2 elements occasionally.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Get current elements
        current_elements = [x for x in range(1, n + 1) if (mask >> x) & 1]
        
        # Shuffle to randomize which element we try to remove first
        random.shuffle(current_elements)
        
        for y in current_elements:
            if time.time() - start_time > time_limit:
                break
                
            # Remove y
            trial_mask = mask & ~(1 << y)
            
            # Greedy add with random order
            order = list(range(1, n + 1))
            random.shuffle(order)
            
            trial_mask = greedy_add(trial_mask, order)
            
            new_size = get_size(trial_mask)
            
            if new_size > best_size:
                best_size = new_size
                best_mask = trial_mask
                mask = trial_mask
                improved = True
                # Restart loop with new set
                break
        
        if not improved:
            # If no improvement with 1-swap, try a 2-swap (remove 2, add many)
            # This is more expensive, so do it less frequently
            if len(current_elements) >= 2:
                # Pick 2 random elements to remove
                y1, y2 = random.sample(current_elements, 2)
                trial_mask = mask & ~(1 << y1) & ~(1 << y2)
                
                order = list(range(1, n + 1))
                random.shuffle(order)
                trial_mask = greedy_add(trial_mask, order)
                
                new_size = get_size(trial_mask)
                if new_size > best_size:
                    best_size = new_size
                    best_mask = trial_mask
                    mask = trial_mask
                    improved = True
            
            if not improved:
                # Try a few random restarts of the greedy phase from scratch?
                # Or just break?
                # Let's try a "kick": remove a random element, don't refill, just break? No.
                # Let's try 5 random restarts of the *entire* greedy process from the Cantor set.
                # This might find a different basin.
                for _ in range(5):
                    if time.time() - start_time > time_limit:
                        break
                    temp_mask = 0
                    # Reconstruct Cantor set
                    for x in range(1, n + 1):
                        num = x
                        valid = True
                        while num > 0:
                            if num % 3 == 2:
                                valid = False
                                break
                            num //= 3
                        if valid:
                            temp_mask |= (1 << x)
                    
                    order = list(range(1, n + 1))
                    random.shuffle(order)
                    temp_mask = greedy_add(temp_mask, order)
                    temp_size = get_size(temp_mask)
                    
                    if temp_size > best_size:
                        best_size = temp_size
                        best_mask = temp_mask
                        mask = temp_mask
                        improved = True
                        break
                
                if not improved:
                    # Stuck. Break to avoid wasting time? 
                    # Or continue with more aggressive 2-swaps?
                    # Let's just break.
                    break

    result = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    return sorted(result)
