import time

def construct():
    start_time = time.time()
    deadline = 90.0 # 90s limit, keep margin
    
    # Initial set: all odd numbers
    S = set(range(1, 257, 2))
    
    # Candidate evens
    candidates = list(range(2, 257, 2))
    
    # Sort candidates? Maybe smaller ones first or arbitrary.
    # Let's try adding them one by one.
    
    def is_compatible(a, b):
        if (a * b) % (a + b) == 0:
            return False
        return True

    # Precompute compatibility with odds?
    # Or just check on the fly.
    
    # To speed up, we can precompute which evens are compatible with ALL odds.
    # If an even is not compatible with some odd, we skip it in the first pass.
    # But we might add it later if we remove that odd? 
    # The task says "Maximize cardinality". 
    # Removing an odd to add an even is a swap.
    # Given the analysis, swaps seem to be 1-for-1 or worse.
    # So keeping all odds is likely optimal or near-optimal.
    # Thus, we only add evens that are compatible with ALL odds.
    
    safe_evens = []
    for e in candidates:
        compatible = True
        for o in S:
            if not is_compatible(e, o):
                compatible = False
                break
        if compatible:
            safe_evens.append(e)
        
        if time.time() > deadline - 5.0:
            break

    # Now, greedily add safe evens to S, checking compatibility with existing S.
    # S initially contains all odds.
    # We add safe evens one by one.
    # Since safe evens are compatible with all odds, we only need to check against previously added evens.
    
    added_evens = []
    for e in safe_evens:
        # Check against added_evens
        ok = True
        for ae in added_evens:
            if not is_compatible(e, ae):
                ok = False
                break
        if ok:
            S.add(e)
            added_evens.append(e)
            
        if time.time() > deadline - 2.0:
            break

    return list(S)
