import time

def construct():
    """
    Find two positive integers a, b with a + b = 123457 that minimize max(P(a), P(b)),
    where P(n) is the largest prime factor of n (and P(1) = 1).
    
    Strategy:
    1. Precompute the largest prime factor (LPF) for every number up to 123457.
    2. Binary search for the minimal threshold K such that there exist a, b 
       with a+b=123457, P(a)<=K, and P(b)<=K.
    3. Use bitmasks to efficiently check for the existence of such a pair.
    4. Extract the pair and return it.
    """
    target = 123457
    start_time = time.time()
    # Safety margin: 60s limit, use 50s
    deadline = start_time + 50.0

    # 1. Precompute largest prime factor
    lpf = [0] * (target + 1)
    lpf[1] = 1
    for i in range(2, target + 1):
        if lpf[i] == 0:
            for j in range(i, target + 1, i):
                lpf[j] = i

    # Helper to check if a valid pair exists for a given K
    def check_k(k):
        if time.time() > deadline:
            return None  # Timeout
        
        # Build bitmask of numbers n such that P(n) <= k
        # Bit i is set if i is K-smooth
        mask = 0
        for n in range(1, target + 1):
            if lpf[n] <= k:
                mask |= (1 << n)
        
        # Build reflected bitmask
        # Bit i is set if (target - i) is K-smooth
        # So if n is K-smooth, we set bit (target - n)
        rmask = 0
        for n in range(1, target):
            if lpf[n] <= k:
                rmask |= (1 << (target - n))
        
        # If there is an intersection, there exists a such that
        # a is K-smooth (bit a in mask) and target-a is K-smooth (bit a in rmask)
        if mask & rmask:
            return True
        return False

    # 2. Binary search for minimal K
    low = 2
    high = target
    best_k = None
    
    while low <= high:
        if time.time() > deadline:
            break
        mid = (low + high) // 2
        res = check_k(mid)
        if res is None:
            break
        if res:
            best_k = mid
            high = mid - 1
        else:
            low = mid + 1

    # Fallback if binary search fails or times out
    if best_k is None:
        # Return a valid pair, e.g., 1 and target-1
        # P(1)=1, P(target-1) is some prime
        return [1, target - 1]

    # 3. Extract the pair
    # Rebuild masks for best_k to find the specific pair
    mask = 0
    for n in range(1, target + 1):
        if lpf[n] <= best_k:
            mask |= (1 << n)
    
    rmask = 0
    for n in range(1, target):
        if lpf[n] <= best_k:
            rmask |= (1 << (target - n))
            
    intersection = mask & rmask
    
    if intersection == 0:
        return [1, target - 1]
        
    # Find the lowest set bit in intersection
    # This corresponds to the smallest 'a'
    # (intersection & -intersection) isolates the lowest set bit
    # bit_length() - 1 gives the index
    a = (intersection & -intersection).bit_length() - 1
    b = target - a
    
    return [a, b]
