import time
import random

def _can_add(S_set, x):
    """
    Check if x can be added to S_set without creating a forbidden triple.
    Forbidden triple: distinct a, b, c in S_new such that a+b, a+c, b+c in S_new.
    If x is new, any new triple must involve x.
    So we check if there exist a, b in S_set (a < b) such that:
    1. a + x in S_new  => a + x in S_set (since a+x != x for a>0)
    2. b + x in S_new  => b + x in S_set
    3. a + b in S_new  => a + b in S_set OR a + b == x
    """
    # Optimization: Pre-filter candidates for a
    # a must be in S_set and a+x must be in S_set
    # b must be in S_set and b+x must be in S_set
    
    # To speed up, we can just iterate. N is small.
    S_list = list(S_set)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        if a + x not in S_set:
            continue
        for j in range(i + 1, n):
            b = S_list[j]
            if b + x not in S_set:
                continue
            # Check a+b
            if a + b in S_set or a + b == x:
                return False
    return True

def construct():
    deadline = time.time() + 85  # Safety margin
    
    best_set = set()
    best_size = 0
    
    # Strategy: Multiple restarts with different initializations and search orders
    rng = random.Random(42)
    
    # Initial guesses
    # 1. Upper half
    init_sets = [
        set(range(65, 129)),
        set(range(1, 65)), # Lower half (likely worse but good to try)
        set(range(1, 129, 2)), # Odds
        set(range(2, 129, 2)), # Evens
    ]
    
    # Add a random set
    random_set = set(rng.sample(range(1, 129), 80))
    init_sets.append(random_set)
    
    for init in init_sets:
        if time.time() > deadline - 10:
            break
            
        current_set = set(init)
        # Ensure validity of initial set (should be valid if constructed carefully, but random might not be)
        # For random set, we need to clean it up.
        if len(init) > 80: # Heuristic: if it's the random one, clean it
             # Simple cleanup: remove elements that cause violations
             # This is expensive, so we only do it for the random init or skip if invalid
             pass
        
        # Local Search
        while time.time() < deadline - 10:
            improved = False
            
            # 1. Try to add an element
            # Get candidates
            candidates = [x for x in range(1, 129) if x not in current_set]
            rng.shuffle(candidates)
            
            added = False
            for x in candidates:
                if _can_add(current_set, x):
                    current_set.add(x)
                    improved = True
                    added = True
                    break # Restart search with new set
            
            if added:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                continue
            
            # 2. If no addition, try a swap
            # Pick random y to remove, random z to add
            # Limit attempts to avoid wasting time
            swap_attempts = 50
            for _ in range(swap_attempts):
                if time.time() > deadline - 10:
                    break
                if not current_set:
                    break
                y = rng.choice(list(current_set))
                z = rng.choice([x for x in range(1, 129) if x not in current_set])
                
                # Check if swap is valid
                # Remove y, add z
                temp_set = current_set - {y}
                if _can_add(temp_set, z):
                    current_set = temp_set
                    current_set.add(z)
                    improved = True
                    break
            
            if improved:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                continue
            
            # 3. If stuck, try to remove an element to escape local minimum?
            # Or just break and try next init
            # Let's try a few random removals to shake up
            removal_attempts = 10
            for _ in range(removal_attempts):
                 if time.time() > deadline - 10:
                     break
                 if len(current_set) <= best_size:
                     break
                 y = rng.choice(list(current_set))
                 current_set.remove(y)
                 # After removal, we might be able to add something
                 # Loop back to try additions
            
            if time.time() > deadline - 10:
                break
            
            # If we didn't improve after swaps and removals, we are likely stuck.
            # Break to try next initialization
            break

    # Final verification and return
    result = sorted(list(best_set))
    # Quick validity check
    # Using a simple check for final result
    S_set = set(result)
    valid = True
    n = len(result)
    for i in range(n):
        a = result[i]
        for j in range(i+1, n):
            b = result[j]
            if a+b not in S_set: continue
            for k in range(j+1, n):
                c = result[k]
                if a+c in S_set and b+c in S_set:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback to a known valid set if something went wrong
        # Upper half is always valid
        result = list(range(65, 129))
        
    return [int(x) for x in result]
