import time

def construct():
    p = 1009
    N = 1008
    
    # Precompute modular inverses
    # inv[x] is the unique y in [1, 1008] such that x*y % p == 1
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    best_len = 0
    best_start = 1
    
    # Deadline guard
    start_time = time.time()
    deadline = 80.0 # 80 seconds limit, plenty of margin
    
    # Iterate over all possible starting positions
    for s in range(1, p):
        if time.time() > deadline:
            break
            
        seen = set()
        max_len_for_s = 0
        
        # Extend the interval as far as possible
        for e in range(s, p):
            y = inv[e]
            # If the inverse of e is already in the interval, we cannot include e
            if y in seen:
                break
            
            seen.add(e)
            current_len = e - s + 1
            if current_len > max_len_for_s:
                max_len_for_s = current_len
        
        if max_len_for_s > best_len:
            best_len = max_len_for_s
            best_start = s
            
    return [best_start, best_len]
