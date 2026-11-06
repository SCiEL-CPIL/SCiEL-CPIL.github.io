<<<<<<< SEARCH
def _local_search(selected, deadline):
    """Local search with add, swap, and kick operations."""
    S = set(selected)
    best_S = set(S)
    best_sum = _reciprocal_sum(S)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Add any missing element that fits
        for x in range(1, 257):
            if x not in S and _can_add(S, x):
                S.add(x)
                improved = True
                new_sum = _reciprocal_sum(S)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_S = set(S)
        
        # Phase 2: Try swaps - remove y, add x where x < y (so 1/x > 1/y)
        swap_improved = True
        while swap_improved and time.time() < deadline:
            swap_improved = False
            best_gain = 0
            best_x = None
            best_y = None
            for x in range(1, 257):
                if x in S:
                    continue
                for y in list(S):
                    if y <= x:
                        continue
                    gain = 1.0 / x - 1.0 / y
                    if gain <= best_gain:
                        continue
                    # Check if (S - {y}) U {x} is valid
                    S_test = S - {y}
                    if _can_add(S_test, x):
                        best_gain = gain
                        best_x = x
                        best_y = y
            if best_x is not None:
                S.discard(best_y)
                S.add(best_x)
                swap_improved = True
                new_sum = _reciprocal_sum(S)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_S = set(S)
                improved = True
        
        # Phase 3: If no improvement, try a kick
        if not improved:
            # Random kick: remove 3-7 random elements, then re-add greedily
            if len(S) > 15:
                kick_size = random.randint(3, 7)
                to_remove = random.sample(list(S), min(kick_size, len(S) - 10))
                for r in to_remove:
                    S.discard(r)
                # Re-add greedily
                for x in range(1, 257):
                    if x not in S and _can_add(S, x):
                        S.add(x)
                new_sum = _reciprocal_sum(S)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_S = set(S)
    
    return best_S
=======
def _local_search(selected, deadline):
    """Local search with add, swap, and kick operations."""
    S = set(selected)
    best_S = set(S)
    best_sum = _reciprocal_sum(S)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Add any missing element that fits
        # Iterate x from 1 to 256, prefer small x (high reciprocal)
        for x in range(1, 257):
            if x not in S and _can_add(S, x):
                S.add(x)
                improved = True
                new_sum = _reciprocal_sum(S)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_S = set(S)
        
        # Phase 2: Try swaps - First Improvement strategy
        # Iterate x (to add) from 1 to 256, y (to remove) from 256 down to 1
        # This prioritizes high gain swaps (small x, large y)
        swap_found = True
        while swap_found and time.time() < deadline:
            swap_found = False
            for x in range(1, 257):
                if x in S:
                    continue
                # We want to remove y > x to ensure gain > 0
                # Iterate y from 256 down to x+1
                for y in range(256, x, -1):
                    if y not in S:
                        continue
                    # Check if (S - {y}) U {x} is valid
                    # Optimization: Check if x conflicts with S\{y}
                    # x conflicts with a in S\{y} if 2x-a in S\{y} or 2a-x in S\{y}
                    # We can just use _can_add on a temporary set, but creating set is slow.
                    # Let's check manually to avoid set creation.
                    valid = True
                    for a in S:
                        if a == y:
                            continue
                        # Check if a, x, b or b, a, x or a, b, x forms AP
                        # 1. x is middle: a, x, b -> b = 2x - a. If b in S and b != y, invalid.
                        b = 2 * x - a
                        if b != a and b != x and b in S and b != y:
                            valid = False
                            break
                        # 2. x is last: b, a, x -> b = 2a - x. If b in S and b != y, invalid.
                        b = 2 * a - x
                        if b != a and b != x and b in S and b != y:
                            valid = False
                            break
                        # 3. x is first: x, a, b -> b = 2a - x. (Same as 2)
                        # Note: The condition "nonconstant" means distinct terms.
                        # If b == a, it's constant, allowed.
                        # If b == x, it's constant, allowed.
                        # We handled b != a and b != x.
                    
                    if valid:
                        # Perform swap
                        S.discard(y)
                        S.add(x)
                        swap_found = True
                        improved = True
                        new_sum = _reciprocal_sum(S)
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_S = set(S)
                        break # Break y loop, continue x loop? Or break both?
                        # Usually first improvement means we restart the search or continue.
                        # Let's restart the swap phase to find more improvements.
                        # Breaking out of both loops is safer for "First Improvement" restart.
                        # But we are in a while loop that restarts. So we can just break inner and outer.
                        # Let's use a flag or just break and let the while loop restart.
                        # Actually, breaking the `for y` loop and then the `for x` loop is good.
                        # We can just `return`? No, we want to continue adding.
                        # Let's break out of the `for x` loop too.
                    else:
                        continue
                if swap_found:
                    break # Break x loop
            
        # Phase 3: If no improvement, try a kick
        if not improved:
            # Random kick: remove 3-7 random elements, then re-add greedily
            if len(S) > 15:
                kick_size = random.randint(3, 7)
                to_remove = random.sample(list(S), min(kick_size, len(S) - 10))
                for r in to_remove:
                    S.discard(r)
                # Re-add greedily
                for x in range(1, 257):
                    if x not in S and _can_add(S, x):
                        S.add(x)
                new_sum = _reciprocal_sum(S)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_S = set(S)
    
    return best_S
>>>>>>> REPLACE

Additionally, I will adjust the deadline in `construct` to be safer (80s instead of 85s) and ensure the greedy initialization is robust.

