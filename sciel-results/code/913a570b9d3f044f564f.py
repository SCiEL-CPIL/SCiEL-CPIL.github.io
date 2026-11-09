import time
import random

N = 512

def construct():
    start_time = time.time()
    timeout = 85.0
    
    def can_add(mask, x):
        # Returns True if x can be added to mask without creating a 4-AP
        # x is 1-based index. Bit x-1? No, let's use bit x for simplicity, bit 0 is unused.
        
        # 1. x, x+d, x+2d, x+3d
        # d max: (512-x)//3
        for d in range(1, (512 - x) // 3 + 1):
            if (mask & (1 << (x + d))) and (mask & (1 << (x + 2 * d))) and (mask & (1 << (x + 3 * d))):
                return False
        
        # 2. x-d, x, x+d, x+2d
        max_d = min(x - 1, (512 - x) // 2)
        for d in range(1, max_d + 1):
            if (mask & (1 << (x - d))) and (mask & (1 << (x + d))) and (mask & (1 << (x + 2 * d))):
                return False
        
        # 3. x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, 512 - x)
        for d in range(1, max_d + 1):
            if (mask & (1 << (x - 2 * d))) and (mask & (1 << (x - d))) and (mask & (1 << (x + d))):
                return False
        
        # 4. x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (mask & (1 << (x - 3 * d))) and (mask & (1 << (x - 2 * d))) and (mask & (1 << (x - d))):
                return False
        
        return True

    def greedy_fill(S, mask):
        # Add all possible elements
        for x in range(1, N + 1):
            if x not in S and can_add(mask, x):
                S.add(x)
                mask |= (1 << x)
        return S, mask

    # Initialize with a few greedy strategies
    best_S = set()
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Ascending
    S1 = set()
    M1 = 0
    for x in range(1, N + 1):
        if can_add(M1, x):
            S1.add(x)
            M1 |= (1 << x)
    if len(S1) > best_size:
        best_S, best_mask, best_size = S1, M1, len(S1)
        
    # Strategy 2: Descending
    S2 = set()
    M2 = 0
    for x in range(N, 0, -1):
        if can_add(M2, x):
            S2.add(x)
            M2 |= (1 << x)
    if len(S2) > best_size:
        best_S, best_mask, best_size = S2, M2, len(S2)

    # Local Search
    current_S = set(best_S)
    current_mask = best_mask
    current_size = best_size
    
    # We will try to improve current_S
    # If stuck, we might restart from a random subset or just try random removals
    
    iterations = 0
    while (time.time() - start_time) < timeout:
        iterations += 1
        
        # 1. Greedy Fill
        current_S, current_mask = greedy_fill(current_S, current_mask)
        current_size = len(current_S)
        
        if current_size > best_size:
            best_S = set(current_S)
            best_mask = current_mask
            best_size = current_size
            
        # 2. Try to improve by removing one element and refilling
        # To avoid getting stuck in the same local optimum, we can pick a random element to remove
        # or iterate through all elements.
        # Iterating through all is expensive. Let's pick a random one.
        
        if not current_S:
            break
            
        # Pick a random element to remove
        # Or, to be more systematic, we can try removing each element in a random order
        # but stop if we find an improvement.
        
        removed_improvement = False
        # Shuffle a copy of the list to try removals in random order
        elements_to_try = list(current_S)
        random.shuffle(elements_to_try)
        
        for x in elements_to_try:
            if (time.time() - start_time) > timeout - 2:
                break
                
            # Save state
            saved_S = set(current_S)
            saved_mask = current_mask
            
            # Remove x
            current_S.remove(x)
            current_mask &= ~(1 << x)
            
            # Refill
            current_S, current_mask = greedy_fill(current_S, current_mask)
            new_size = len(current_S)
            
            if new_size > current_size:
                # Accept improvement
                current_size = new_size
                if current_size > best_size:
                    best_S = set(current_S)
                    best_mask = current_mask
                    best_size = current_size
                removed_improvement = True
                break # Restart the outer loop with the new state
            
            # If no improvement, revert
            current_S = saved_S
            current_mask = saved_mask
            current_size = len(current_S)
            
        if not removed_improvement:
            # If we tried all (or many) removals and didn't improve, we are stuck.
            # We can try a random restart or just continue with random removals.
            # To escape, we can remove a random element and keep the state even if size decreases?
            # No, that's risky. 
            # Let's just continue the loop. The random shuffle will pick different elements next time.
            # But if we always revert, we will keep trying the same set of removals?
            # No, `elements_to_try` is shuffled each time.
            # However, if no single removal leads to an improvement, we are in a local optimum.
            # We might need to accept a worse state to escape (Simulated Annealing).
            # Let's implement a simple SA move:
            # With some probability, accept a move that decreases size by 1.
            
            if random.random() < 0.1: # 10% chance to accept a worse move
                x = random.choice(list(current_S))
                saved_S = set(current_S)
                saved_mask = current_mask
                
                current_S.remove(x)
                current_mask &= ~(1 << x)
                
                # Refill
                current_S, current_mask = greedy_fill(current_S, current_mask)
                new_size = len(current_S)
                
                # Accept if it's not too bad? Or just accept?
                # If we accept, we might lose progress. 
                # But it helps escape local optima.
                # Let's accept if new_size >= current_size - 1
                if new_size >= current_size - 1:
                    current_size = new_size
                    # Do not update best_S unless it's better
                    # But we should track the "current" state for further search
                else:
                    # Revert if it's too bad
                    current_S = saved_S
                    current_mask = saved_mask
                    current_size = len(current_S)
            
    return sorted(best_S)
