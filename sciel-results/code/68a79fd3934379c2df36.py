import time
import random

def construct():
    start_time = time.time()
    n = 512
    
    # Precompute all 4-APs to speed up validity checks
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # We need to check if adding x creates a 4-AP.
    # x can be part of a 4-AP in 4 ways:
    # 1. x is a: {x, x+d, x+2d, x+3d}
    # 2. x is a+d: {x-d, x, x+d, x+2d}
    # 3. x is a+2d: {x-2d, x-d, x, x+d}
    # 4. x is a+3d: {x-3d, x-2d, x-d, x}
    
    def is_valid_addition(x, current_set):
        # Check if adding x creates any 4-AP
        # We only need to check APs that include x and at least 2 other elements from current_set?
        # No, we need to check if {x, y, z, w} subset of current_set U {x} is a 4-AP.
        # Since current_set is valid, any new 4-AP must contain x.
        # So we check if there exist y, z, w in current_set such that {x, y, z, w} is a 4-AP.
        
        # Case 1: x is the first term (a)
        # y = x+d, z = x+2d, w = x+3d
        # d must be such that x+3d <= n
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d) in current_set and (x + 2*d) in current_set and (x + 3*d) in current_set:
                return False
        
        # Case 2: x is the second term (a+d)
        # y = x-d, z = x+d, w = x+2d
        # a = x-d >= 1 => d <= x-1
        # w = x+2d <= n => 2d <= n-x => d <= (n-x)//2
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d) in current_set and (x + d) in current_set and (x + 2*d) in current_set:
                return False
        
        # Case 3: x is the third term (a+2d)
        # y = x-2d, z = x-d, w = x+d
        # a = x-2d >= 1 => 2d <= x-1 => d <= (x-1)//2
        # w = x+d <= n => d <= n-x
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            if (x - 2*d) in current_set and (x - d) in current_set and (x + d) in current_set:
                return False
        
        # Case 4: x is the fourth term (a+3d)
        # y = x-3d, z = x-2d, w = x-d
        # a = x-3d >= 1 => 3d <= x-1 => d <= (x-1)//3
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3*d) in current_set and (x - 2*d) in current_set and (x - d) in current_set:
                return False
        
        return True

    # Initial random valid set construction (greedy with random order)
    # This ensures we start with a valid set.
    elements = list(range(1, n + 1))
    random.shuffle(elements)
    current_set = set()
    for x in elements:
        if is_valid_addition(x, current_set):
            current_set.add(x)
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Local Search / Simulated Annealing
    # We will try to improve the set size.
    # Since we want to maximize size, we only accept moves that increase size or are neutral (if we want to move around).
    # Actually, pure hill climbing on size:
    # Move 1: Add an element x not in set. If valid, accept.
    # Move 2: Remove an element y from set, then try to add x. If size increases (remove 1 add 1 = 0 change? No, remove 1, add 1 is 0 change).
    # To increase size, we must find an x such that adding x is valid.
    # If we can't add any x, we are stuck.
    # We can try swaps: remove y, add x. If size is same, we might move to a different local optimum.
    # But to strictly increase size, we need to find a "free" spot.
    
    # Strategy:
    # 1. Try to add any valid element.
    # 2. If no additions possible, try random swaps to rearrange the set to free up space for additions.
    
    current_size = len(current_set)
    no_improvement_count = 0
    max_no_improvement = 500 # Restart or stop if stuck too long
    
    while time.time() - start_time < 50.0: # Safety margin
        improved = False
        
        # 1. Attempt to add elements
        # Try all elements not in set
        for x in range(1, n + 1):
            if x not in current_set:
                if is_valid_addition(x, current_set):
                    current_set.add(x)
                    current_size += 1
                    improved = True
                    # Restart loop to check for more additions
                    break
        
        if improved:
            if current_size > best_size:
                best_set = set(current_set)
                best_size = current_size
            no_improvement_count = 0
            continue
        
        # 2. No additions possible. Try swaps to escape local optimum.
        # Pick a random element y from current_set to remove.
        # Pick a random element x not in current_set to add.
        # Check if {current_set - {y} + {x}} is valid.
        # If valid, accept the swap (size remains same, but structure changes).
        
        # To make this efficient, we can limit the number of swap attempts per "stuck" phase.
        swap_attempts = 0
        max_swaps_per_phase = 100
        
        while swap_attempts < max_swaps_per_phase and time.time() - start_time < 50.0:
            swap_attempts += 1
            if not current_set:
                break
            
            y = random.choice(list(current_set))
            x = random.randint(1, n)
            
            if x == y or x in current_set:
                continue
            
            # Check if removing y and adding x results in a valid set
            # This is equivalent to checking if x is valid in (current_set - {y})
            # We can simulate this.
            
            # Optimization: If x is not valid in current_set, it might still be valid if y is removed.
            # But is_valid_addition checks against the whole set.
            # Let's just remove y, check x, and if valid, keep it. If not, put y back.
            
            current_set.remove(y)
            if is_valid_addition(x, current_set):
                current_set.add(x)
                # Swap accepted. Size is same.
                # Check if we can now add something?
                # The outer loop will try to add again.
                break
            else:
                # Swap rejected, restore y
                current_set.add(y)
        
        if swap_attempts >= max_swaps_per_phase and not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Restart with a new random seed to escape deep local optimum
                elements = list(range(1, n + 1))
                random.shuffle(elements)
                current_set = set()
                for x in elements:
                    if is_valid_addition(x, current_set):
                        current_set.add(x)
                current_size = len(current_set)
                no_improvement_count = 0
                # If this restart is worse, keep best
                if current_size < best_size:
                    # It's possible the restart is bad. We could revert to best_set and try again,
                    # but let's just continue searching from this new state.
                    pass

    # Final check: try to add any remaining valid elements
    for x in range(1, n + 1):
        if x not in best_set:
            if is_valid_addition(x, best_set):
                best_set.add(x)
    
    return sorted(best_set)
