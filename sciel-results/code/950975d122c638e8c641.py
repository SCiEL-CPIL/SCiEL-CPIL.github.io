import time

def construct():
    start_time = time.time()
    p = 1009
    max_val = p - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Self-inverse elements: x where x^2 ≡ 1 (mod p), i.e., x = 1 or x = p-1
    self_inv = {1, p - 1}
    
    # Build conflict pairs: for each x in [2, p-2], if inv[x] != x,
    # record the pair (min(x, inv[x]), max(x, inv[x]))
    # We want to find the longest contiguous interval [s, e] in [1, p-1]
    # such that:
    #   1. No self-inverse element is in [s, e]
    #   2. No conflict pair (a, b) with a < b has both a and b in [s, e]
    
    # For each position b, precompute the maximum a such that (a, b) is a conflict pair
    # This lets us quickly check: for a given start s, the interval [s, e] is valid
    # iff for all b in [s+1, e], max_a_for_b[b] < s
    max_a_for_b = [0] * (p + 1)
    for i in range(2, p - 1):  # skip 1 and p-1 (self-inverse)
        j = inv[i]
        if j == i:
            continue
        a = min(i, j)
        b = max(i, j)
        if a > max_a_for_b[b]:
            max_a_for_b[b] = a
    
    best_len = 0
    best_start = 2
    
    # Try all possible starts from 2 to p-2 (since 1 and p-1 are self-inverse,
    # any interval containing them is invalid)
    for s in range(2, p - 1):
        # If even reaching p-2 can't beat current best, stop
        if (p - 2) - s + 1 <= best_len:
            break
        
        # Find the maximum e such that [s, e] is valid
        # The interval becomes invalid at the first b >= s+1 where max_a_for_b[b] >= s
        # Also, e cannot exceed p-2 (since p-1 is self-inverse)
        max_e = p - 2
        valid_e = max_e
        
        for b in range(s + 1, max_e + 1):
            if max_a_for_b[b] >= s:
                valid_e = b - 1
                break
        
        # Also check: does the interval [s, valid_e] contain any self-inverse?
        # Since s >= 2 and valid_e <= p-2, no self-inverse is in the interval.
        
        length = valid_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
        
        # Time guard
        if time.time() - start_time > 80:
            break
    
    return [best_start, best_len]
