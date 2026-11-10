import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    
    n = 1008
    mod = 1009
    
    # Precompute modular inverses
    inv = [0] * (mod + 1)
    for x in range(1, mod):
        inv[x] = pow(x, mod - 2, mod)
    
    best_start = 1
    best_length = 0
    
    for s in range(1, mod):  # s from 1 to 1008
        if time.time() > deadline:
            break
        
        # Extend interval from s as far as possible
        in_interval = [False] * (mod + 1)
        e = s - 1
        valid = True
        
        while e + 1 < mod:  # next element would be e+1, must be <= 1008
            next_val = e + 1
            if next_val > 1008:
                break
            
            # Check if adding next_val creates a conflict
            # Conflict if inv[next_val] is already in interval, or inv[next_val] == next_val (self-inverse)
            inv_next = inv[next_val]
            
            if in_interval[inv_next] or inv_next == next_val:
                # Adding next_val would create a conflict
                break
            
            # Add next_val to interval
            in_interval[next_val] = True
            e = next_val
        
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s
    
    return [best_start, best_length]
