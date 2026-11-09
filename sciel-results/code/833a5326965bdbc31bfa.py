import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008
    
    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)
        
    # The valid range for start and end is [2, 1007]
    # Because 1 and 1008 are self-inverse and thus forbidden.
    
    def is_valid_interval(start, length):
        end = start + length - 1
        if start < 2 or end > 1007:
            return False
        
        # Check for any x in [start, end] such that inv[x] is also in [start, end]
        # Note: if x == inv[x], it's 1 or 1008, which are excluded.
        # So we are checking for distinct pairs or just general presence.
        # If x is in interval, and inv[x] is in interval, it's invalid.
        for x in range(start, end + 1):
            ix = inv[x]
            if start <= ix <= end:
                return False
        return True

    # Binary search for the maximum length
    # Min length is 1, Max length is 1006 (1007-2+1)
    lo = 1
    hi = 1006
    best_len = 1
    best_start = 2
    
    while lo <= hi:
        mid = (lo + hi) // 2
        found = False
        found_start = -1
        
        # Check if there exists a valid interval of length mid
        for s in range(2, 1008 - mid + 1):
            if is_valid_interval(s, mid):
                found = True
                found_start = s
                break
        
        if found:
            best_len = mid
            best_start = found_start
            lo = mid + 1
        else:
            hi = mid - 1
            
        # Time check
        if time.time() - start_time > 80:
            break

    return [best_start, best_len]
