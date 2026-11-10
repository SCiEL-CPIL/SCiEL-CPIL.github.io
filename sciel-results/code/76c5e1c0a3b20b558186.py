import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, keep 10s margin
    
    n = 256
    
    # 1. Greedy Initialization
    selected = set()
    for x in range(1, n + 1):
        # Check if x forms a 3-AP with any a, b in selected
        # Since x is the largest, we only check x = 2b - a
        valid = True
        for a in selected:
            for b in selected:
                if a < b and 2 * b - a == x:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.add(x)
    
    def get_weight(s):
        return sum(1.0 / x for x in s)
    
    current_weight = get_weight(selected)
    best_set = set(selected)
    best_weight = current_weight
    
    # Helper to check if x can be added to set S
    def can_add(x, S):
        # Check if x forms 3-AP with any a, b in S
        # x can be smallest, middle, or largest
        for a in S:
            for b in S:
                if a == b: continue
                # Check all 3 permutations of (x, a, b) forming AP
                # 1. x, a, b -> 2a = x + b
                if 2 * a == x + b:
                    return False
                # 2. a, x, b -> 2x = a + b
                if 2 * x == a + b:
                    return False
                # 3. a, b, x -> 2b = a + x
                if 2 * b == a + x:
                    return False
        return True

    # Helper to check if S is valid (should always be true in our moves)
    # Not needed if we only move to valid states.

    # Local Search / Simulated Annealing
    # We will try swaps: remove y, add x
    
    # Precompute list of non-selected elements
    non_selected = [x for x in range(1, n + 1) if x not in selected]
    
    # To speed up, we can sort non_selected by weight (1/x) descending
    # So we try adding small numbers first
    
    t = 1.0
    cooling_rate = 0.999
    min_t = 1e-4
    
    while time.time() < deadline:
        if t < min_t:
            break
            
        # Generate a random move
        # Bias towards swapping with high-weight x
        # Pick a random y from selected
        # Pick a random x from non_selected (weighted by 1/x? or just uniform)
        
        # Let's try a few random swaps and pick the best improving one if found
        # Or just Simulated Annealing steps
        
        y = random.choice(list(selected))
        # Choose x. We prefer small x.
        # Pick a random index, but maybe bias?
        # Let's just pick random for now, SA will handle it.
        x = random.choice(non_selected)
        
        if y == x:
            continue
            
        # Check validity of S' = S - {y} + {x}
        # We only need to check if x is compatible with S - {y}
        # Create a temporary set view?
        # S_new = selected - {y}
        # Check can_add(x, S_new)
        
        # Optimization: can_add(x, S) checks against all S.
        # We need to check against S \ {y}.
        # If can_add(x, S) is True, then can_add(x, S\{y}) is True.
        # If can_add(x, S) is False, it might be True for S\{y} if the conflict involved y.
        
        # So we check:
        valid_new = True
        # Iterate through S, skipping y
        for a in selected:
            if a == y:
                continue
            for b in selected:
                if b == y or b == a:
                    continue
                # Check 3-AP with x, a, b
                if 2 * a == x + b:
                    valid_new = False
                    break
                if 2 * x == a + b:
                    valid_new = False
                    break
                if 2 * b == a + x:
                    valid_new = False
                    break
            if not valid_new:
                break
        
        if valid_new:
            new_weight = current_weight - 1.0 / y + 1.0 / x
            delta = new_weight - current_weight
            
            # Acceptance probability
            if delta > 0 or random.random() < pow(2.71828, delta / t):
                # Apply move
                selected.remove(y)
                selected.add(x)
                
                # Update non_selected list
                # Remove x, add y
                # This is O(N). N=256. Acceptable.
                non_selected.remove(x)
                non_selected.append(y)
                
                current_weight = new_weight
                
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_set = set(selected)
        
        t *= cooling_rate
        
    return list(best_set)
