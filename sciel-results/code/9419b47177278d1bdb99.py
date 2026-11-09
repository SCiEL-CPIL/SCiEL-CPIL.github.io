<<<<<<< SEARCH
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if k == 3 and g == 16:
        # Try LUW with increasing d until we find a valid graph
        # Minimum d for girth 16?
        # Girth is roughly 2d. So d ~ 8.
        # Let's try d from 7 to 12.
        for d in range(7, 13):
            if time.time() > deadline - 10:
                break
            result = _construct_luw(k, g, d)
            if result is not None:
                return result
        
        # Fallback to original large d if all fail (should not happen if girth logic is correct)
        # But just in case, use a large d that is guaranteed to work?
        # The original code used d=11.
        result = _construct_luw(k, g, 11)
        if result is not None:
            return result
            
    # Generic fallback
    # If k, g are not 3, 16, use the original logic or a simple LUW
    # For now, just use LUW with a reasonable d
    d = g // 2 + 2
    result = _construct_luw(k, g, d)
    if result is not None:
        return result
    
    # If all fails, return a trivial graph? No, must be valid.
    # The task implies we should always find one.
    # We'll raise an exception if we can't, but we should be safe.
    raise Exception("Failed to construct graph")
=======
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    if k == 3 and g == 16:
        # LUW girth is 2d, so d=8 gives girth 16
        # Try d=8 first (smallest valid), then larger if needed
        for d in range(8, 13):
            if time.time() > deadline - 10:
                break
            result = _construct_luw(k, g, d)
            if result is not None:
                return result
        
        # Fallback
        result = _construct_luw(k, g, 11)
        if result is not None:
            return result
            
    # Generic fallback
    d = max(g // 2, 4)
    result = _construct_luw(k, g, d)
    if result is not None:
        return result
    
    raise Exception("Failed to construct graph")
>>>>>>> REPLACE
