"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    # inv[i] is the modular inverse of i mod p
    inv = [0] * (p + 1)
    for i in range(1, p):
        # Since p is small, we can just brute force or use pow
        # pow(i, p-2, p) is efficient
        inv[i] = pow(i, p - 2, p)
        
    # Two-pointer sliding window
    left = 1
    in_window = set()
    max_len = 0
    best_start = 1
    
    for right in range(1, n + 1):
        if time.time() > deadline:
            break
            
        inv_r = inv[right]
        is_self_inv = (right == 1 or right == 1008)
        
        if is_self_inv:
            # right cannot be in the interval
            # Current valid interval is [left, right-1]
            curr_len = right - 1 - left + 1
            if curr_len > max_len:
                max_len = curr_len
                best_start = left
        else:
            # Shrink window until inv_r is not in window
            while inv_r in in_window:
                in_window.remove(left)
                left += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
                
            in_window.add(right)
            # Current valid interval is [left, right]
            curr_len = right - left + 1
            if curr_len > max_len:
                max_len = curr_len
                best_start = left
                
    # Ensure we return a valid non-empty interval
    # If max_len is 0, we need to find any valid single element
    # Single elements are valid unless they are self-inverse (1, 1008)
    if max_len == 0:
        for i in range(1, n + 1):
            if i != 1 and i != 1008:
                return [i, 1]
        # If all are self-inverse (impossible for p=1009), fallback
        return [2, 1]
        
    return [best_start, max_len]
# EVOLVE-BLOCK-END
