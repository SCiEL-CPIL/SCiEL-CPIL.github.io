import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s limit
    LIMIT = 1009
    MAX_VAL = 1008
    
    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)
        
    best_len = 0
    best_start = 1
    s = 1
    
    # Iterate through each potential end of the interval
    for e in range(1, MAX_VAL + 1):
        partner = inv[e]
        
        # Check for conflict with the new element e
        if partner == e:
            # Self-inverse: e cannot be in any valid interval
            s = e + 1
        else:
            # If the partner is currently in the window [s, e-1], 
            # we must move s past the partner.
            if s <= partner < e:
                s = partner + 1
        
        # Calculate current valid interval length
        current_len = e - s + 1
        
        if current_len > best_len:
            best_len = current_len
            best_start = s
            
        # Time check with safety margin
        if time.time() - start_time > DEADLINE:
            break
            
    return [best_start, best_len]
