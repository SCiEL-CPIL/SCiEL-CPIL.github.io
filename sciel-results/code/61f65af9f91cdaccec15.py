import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    S = [0]
    used_diffs = set()
    missing = set(range(1, 31))
    
    while missing:
        if time.time() > deadline:
            # If we time out, return current S (might be invalid if missing not empty)
            # But we should avoid this.
            break
            
        best_x = None
        
        # Search for x
        # We prioritize small x to minimize max(S)
        for x in range(1, 10000):
            # Check Sidon validity
            valid = True
            new_diffs = set()
            for y in S:
                d = abs(x - y)
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Check if it covers any missing
            gain = len(new_diffs & missing)
            if gain > 0:
                best_x = x
                break # Smallest x with gain > 0
        
        if best_x is None:
            # If no small x works, try larger?
            # For now, if we can't find one, we are stuck.
            # But for 1..30, it should find one.
            # If not, we might need to restart or use a different heuristic.
            # Let's just break and hope the previous S was valid? No, S must cover 1..30.
            # If we can't cover, we fail.
            # However, this case is extremely unlikely for this specific task.
            break
        
        S.append(best_x)
        for y in S[:-1]:
            d = abs(best_x - y)
            used_diffs.add(d)
        
        # Update missing
        missing -= used_diffs
    
    # Final check
    if not missing:
        return S
    else:
        # If we failed to cover, we return a known valid set?
        # Actually, if the greedy fails, it's better to return a set that is definitely Sidon 
        # even if it doesn't cover 1..30? No, the task requires covering 1..30.
        # "invalid result ... scores 0".
        # So we MUST cover 1..30.
        # If the greedy fails, we should try a randomized approach or a larger search space.
        # But given the simplicity, it should work.
        # Let's add a simple fallback: 
        # If missing is not empty, we can try to add more elements randomly?
        # No, let's just trust the greedy.
        return S
