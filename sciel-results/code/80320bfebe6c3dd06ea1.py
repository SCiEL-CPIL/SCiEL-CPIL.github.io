import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fixed elements required in the set
    fixed = {1, 2, 5, 11}
    
    # Initial state
    S = list(fixed)
    
    # Calculate sums
    def calc_sums(S):
        sums = set()
        for i, x in enumerate(S):
            for y in S[i:]:
                sums.add(x + y)
        return sums

    sums = calc_sums(S)
    best_S = S[:]
    best_size = len(S)
    
    # Candidates to add
    candidates = list(range(12, 285))
    
    # Helper: Check if x can be added to S
    def can_add(x, S, sums):
        if x in S:
            return False
        if 2 * x in sums:
            return False
        for y in S:
            if x + y in sums:
                return False
        return True

    # Helper: Add x to S and update sums
    def do_add(x, S, sums):
        for y in S:
            sums.add(x + y)
        sums.add(2 * x)
        S.append(x)

    # Helper: Remove x from S and update sums
    def do_remove(x, S, sums):
        for y in S:
            if y != x:
                sums.discard(x + y)
        sums.discard(2 * x)
        S.remove(x)

    # Main search loop
    while time.time() < deadline:
        # Phase 1: Greedy Add
        # Try to add any valid candidate
        # We shuffle candidates to explore different paths
        random.shuffle(candidates)
        added_any = False
        for x in candidates:
            if can_add(x, S, sums):
                do_add(x, S, sums)
                added_any = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S[:]
                # Don't break, try to add more in this iteration
                # But we need to re-evaluate other candidates against the new S
                # Since we are iterating over a static list, and S changes,
                # a candidate that was invalid before might become valid?
                # No, adding elements only adds sums, making it *harder* to add others.
                # So if a candidate was invalid, it remains invalid.
                # If it was valid, we added it.
                # So we can just continue the loop.
        
        if not added_any:
            # Phase 2: Local Search / Rearrangement
            # We are stuck. Try to change the structure to allow more additions.
            
            # Get non-fixed elements
            non_fixed = [x for x in S if x not in fixed]
            
            if not non_fixed:
                break # Cannot improve
            
            # Try "Remove 1, Add 2" strategy
            # This increases size by 1 if successful
            success = False
            for _ in range(100): # Limit attempts per stuck phase
                if time.time() > deadline:
                    break
                
                # Pick a random non-fixed element to remove
                z = random.choice(non_fixed)
                
                # Backup state
                S_backup = S[:]
                sums_backup = sums.copy()
                
                # Remove z
                do_remove(z, S, sums)
                
                # Try to add two new elements
                # We prioritize candidates that are "close" to z or just random
                # Using random shuffle for simplicity
                random.shuffle(candidates)
                
                added_count = 0
                for w in candidates:
                    if w == z:
                        continue
                    if can_add(w, S, sums):
                        do_add(w, S, sums)
                        added_count += 1
                        if added_count == 2:
                            break
                
                if added_count == 2:
                    # We successfully swapped 1 for 2, net gain +1
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S[:]
                    success = True
                    break # Break out of rearrangement attempts, go back to Add phase
                else:
                    # Revert
                    S = S_backup
                    sums = sums_backup
                    # If we fail many times, we might be in a very tight local optimum
                    # We could try "Remove 2, Add 2" or "Remove 1, Add 1" (swap)
                    # But Remove 1 Add 2 is the most direct path to growth.
            
            if not success:
                # If we couldn't grow, we might try simple swaps to escape
                # Or just break and rely on the fact that we've explored a lot?
                # Let's try a few simple swaps (Remove 1, Add 1) to change the landscape
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    z = random.choice(non_fixed)
                    S_backup = S[:]
                    sums_backup = sums.copy()
                    
                    do_remove(z, S, sums)
                    
                    random.shuffle(candidates)
                    swapped = False
                    for w in candidates:
                        if w == z:
                            continue
                        if can_add(w, S, sums):
                            do_add(w, S, sums)
                            swapped = True
                            break
                    
                    if swapped:
                        # We maintained size, but changed structure
                        # Continue to next iteration of main loop to try adding
                        break
                    else:
                        S = S_backup
                        sums = sums_backup

    return best_S
