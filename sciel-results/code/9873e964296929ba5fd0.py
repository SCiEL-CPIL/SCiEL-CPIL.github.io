import numpy as np

def construct():
    p = 1009
    # Compute inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_len = 0
    
    # Self-inverse elements are 1 and 1008. 
    # Valid intervals cannot contain 1 or 1008.
    # So we search in [2, 1007].
    
    for s in range(2, 1008):
        forbidden = set()
        current_len = 0
        # We start trying to build an interval from s
        for e in range(s, 1008):
            # If e is forbidden, we cannot include it
            if e in forbidden:
                break
            
            # We can include e
            current_len += 1
            
            # The inverse of e becomes forbidden for future additions
            inv_e = inv[e]
            forbidden.add(inv_e)
            
            # Update best
            if current_len > best_len:
                best_len = current_len
                best_start = s
        
    return [best_start, best_len]
