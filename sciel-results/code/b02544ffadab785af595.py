import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # 1. Start with all odd numbers
    S = set(range(1, 257, 2))
    
    # 2. Identify even numbers that don't conflict with any odd
    safe_evens = []
    for b in range(2, 257, 2):
        compatible = True
        for a in S:
            if (a * b) % (a + b) == 0:
                compatible = False
                break
        if compatible:
            safe_evens.append(b)
    
    # 3. Add safe evens greedily, checking conflicts with existing set
    # We might want to be smarter, but greedy is fast.
    # To maximize, maybe try different orders?
    # For now, just add them if they don't conflict with current S.
    # But safe_evens are already compatible with odds.
    # We just need to check conflicts among the evens we add.
    
    added_evens = []
    for b in safe_evens:
        # Check against odds (already known to be fine)
        # Check against added_evens
        conflict = False
        for existing in added_evens:
            if (b * existing) % (b + existing) == 0:
                conflict = True
                break
        if not conflict:
            added_evens.append(b)
            S.add(b)
            
    # 4. Local search to improve
    # Try swapping an even in added_evens with one not in it?
    # Or just return the current set.
    
    # Check time
    if time.time() > deadline:
        return list(S)
        
    return list(S)
