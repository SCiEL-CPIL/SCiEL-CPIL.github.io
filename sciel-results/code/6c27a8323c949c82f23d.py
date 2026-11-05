import random
import time

def construct():
    n = 256
    # Start with a known valid set
    # {171, ..., 256} is valid.
    initial_set = set(range(171, n + 1))
    
    S = set(initial_set)
    best_S = set(S)
    best_size = len(S)
    
    def is_valid_with_add(S, x):
        """Check if adding x to S creates any violation."""
        if x in S:
            return True
            
        S_list = sorted(S)
        
        # Case 1: x is a (smallest)
        # b, c in S, b > x, c >= b
        greater = [v for v in S_list if v > x]
        for i, b in enumerate(greater):
            for c in greater[i:]:
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is b (middle)
        # a < x <= c. a, c in S.
        less = [v for v in S_list if v < x]
        greater_or_eq = [v for v in S_list if v >= x]
        
        for a in less:
            for c in greater_or_eq:
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is c (largest)
        # a < b <= x. a, b in S.
        less_or_eq = [v for v in S_list if v <= x]
        
        for i, a in enumerate(less_or_eq):
            for b in less_or_eq[i+1:]:
                if (b + x) % a == 0:
                    return False
        
        return True

    start_time = time.time()
    deadline = start_time + 85
    
    # List of candidates to try adding
    # We prioritize smaller numbers as they are "harder" to place?
    # Or random.
    
    # We will run multiple restarts.
    # Each restart tries to build a set from scratch or improve the current one.
    
    # Strategy: 
    # 1. Start with {171..256}.
    # 2. Greedily add from {1..170}.
    # 3. If stuck, try swaps.
    # 4. If still stuck, restart with a slightly different initial set?
    
    for restart in range(50): # Max restarts
        if time.time() > deadline:
            break
            
        # Reset S to initial or best?
        # Let's start from best_S to try to improve it.
        # Or start from scratch to find a different local optimum.
        # Starting from scratch:
        # Try different lower bounds for the initial set.
        # {171..256} is size 86.
        # {160..256} is size 97. Is it valid?
        # Check a=160. 3a=480. Pairs sum to 480?
        # 224+256=480. Both in {160..256}.
        # So {160..256} is invalid.
        
        # So we stick to {171..256} as a safe base.
        
        S = set(initial_set)
        
        # Shuffle candidates
        candidates = list(range(1, 171))
        random.shuffle(candidates)
        
        modified = True
        while modified:
            modified = False
            for x in candidates:
                if time.time() > deadline:
                    break
                if x in S:
                    continue
                if is_valid_with_add(S, x):
                    S.add(x)
                    modified = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
            
            if time.time() > deadline:
                break
            
            # If no addition was made, try swaps
            if not modified:
                # Try a few random swaps
                swapped = False
                for _ in range(5):
                    if not S:
                        break
                    y = random.choice(list(S))
                    # Find a z to swap with
                    # Try random z
                    for _ in range(10):
                        z = random.randint(1, n)
                        if z in S:
                            continue
                        # Check if z is valid in S - {y}
                        # We need to pass S-{y} to is_valid_with_add
                        # But is_valid_with_add takes S and x.
                        # So we call is_valid_with_add(S - {y}, z)
                        if is_valid_with_add(S - {y}, z):
                            S = (S - {y}) | {z}
                            swapped = True
                            # Update best if size increased? No, size same.
                            # But might enable future additions.
                            break
                    if swapped:
                        break
                
                if not swapped:
                    # Stuck. Break to next restart.
                    break

    return list(best_S)

# EVOLVE-BLOCK-END
