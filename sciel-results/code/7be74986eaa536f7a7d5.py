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

    # Sliding window approach (inspiration 2)
    # The interval must be within [2, 1007] because 1 and 1008 are self-inverse
    left = 2
    seen = set()
    
    for right in range(2, 1008):
        conflict = inv[right]
        # If the inverse of the current element is already in the window,
        # we must shrink the window from the left until that element is removed.
        while conflict in seen:
            seen.remove(left)
            left += 1
        
        seen.add(right)
        current_len = right - left + 1
        
        if current_len > best_len:
            best_len = current_len
            best_start = left
        
        # Time check with safety margin
        if time.time() - start_time > 80:
            break

    return [best_start, best_len]
