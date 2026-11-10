import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
        
    # Identify self-inverses
    self_inv = set()
    for i in range(1, MOD):
        if inv[i] == i:
            self_inv.add(i)
            
    best_start = 1
    best_length = 0
    
    # Iterate over all possible start positions
    for s in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # If start is self-inverse, length is 0
        if s in self_inv:
            continue
            
        # Check if s is valid (it is, since not self-inverse)
        # Now try to extend
        current_length = 1
        
        # We can break early if even the max possible length (N - s + 1) is not better than best_length
        if N - s + 1 <= best_length:
            continue
            
        for y in range(s + 1, N + 1):
            if time.time() > deadline:
                break
                
            if y in self_inv:
                break
                
            # Check if inv[y] is in [s, y-1]
            iy = inv[y]
            if s <= iy < y:
                break
                
            current_length += 1
            
        if current_length > best_length:
            best_length = current_length
            best_start = s
            
    return [best_start, best_length]
