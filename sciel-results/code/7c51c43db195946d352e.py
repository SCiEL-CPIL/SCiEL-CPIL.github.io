import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008
    
    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)
        
    best_len = 0
    best_start = 2
    
    # The interval must be within [2, 1007]
    # 1 and 1008 are self-inverse, so they cannot be in the interval.
    
    S = 2
    E = 2
    
    while S <= 1007:
        # Ensure E is at least S
        if E < S:
            E = S
            
        # Try to extend E
        while E + 1 <= 1007:
            next_val = E + 1
            # Check if adding next_val creates a conflict
            # Conflict if inv[next_val] is in [S, E]
            if S <= inv[next_val] <= E:
                break
            else:
                E = next_val
                
        # Update best
        current_len = E - S + 1
        if current_len > best_len:
            best_len = current_len
            best_start = S
            
        S += 1
        
        # Time check
        if time.time() - start_time > 50:
            break

    return [best_start, best_len]
