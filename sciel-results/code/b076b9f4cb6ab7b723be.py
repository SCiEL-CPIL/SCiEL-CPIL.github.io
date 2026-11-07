def construct():
    import time
    import random
    
    def is_valid(S):
        S_set = set(S)
        items = sorted(S)
        n = len(items)
        max_val = max(S_set)
        for i in range(n):
            a = items[i]
            for j in range(i+1, n):
                b = items[j]
                s_ab = a + b
                if s_ab > max_val:
                    break
                if s_ab not in S_set:
                    continue
                for k in range(j+1, n):
                    c = items[k]
                    s_ac = a + c
                    if s_ac > max_val:
                        break
                    if s_ac not in S_set:
                        continue
                    s_bc = b + c
                    if s_bc in S_set:
                        return False
        return True

    start_time = time.time()
    time_limit = 85.0
    
    best_size = 66
    best_set = list(range(63, 129))
    
    # Try several random restarts
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Start with a random subset of size 65
        S = random.sample(range(1, 129), 65)
        
        # Fix S to be valid
        # While invalid, remove a random element? No, we need to find the conflict.
        # Simpler: If invalid, just discard this restart?
        # Better: Use a "repair" step.
        # Since finding the exact conflict is expensive, we can just try to build it up.
        
        # Alternative: Start with valid set, perturb it.
        # Let's stick to the greedy addition from the known valid set.
        # But add a "shake" step: remove a random element, then try to add 2 random elements.
        
        current = best_set[:]
        # Remove 1 random
        if current:
            current.remove(random.choice(current))
        
        # Try to add back
        for _ in range(10):
            if time.time() - start_time > time_limit:
                break
            candidates = [x for x in range(1, 129) if x not in current]
            if not candidates:
                break
            x = random.choice(candidates)
            new_set = current + [x]
            if is_valid(new_set):
                current = new_set
                if len(current) > len(best_set):
                    best_set = current[:]
                    best_size = len(current)
                    
    return best_set
