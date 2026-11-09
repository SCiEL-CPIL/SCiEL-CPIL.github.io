import numpy as np

def construct():
    max_n = 1000000
    max_val = max_n + 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # 1. Generate all 23-smooth numbers <= max_val
    # A number is 23-smooth if all its prime factors are in primes.
    # We can generate them by multiplying primes.
    smooths = set([1])
    # Simple generation:
    # Start with 1.
    # For each prime p, multiply existing smooths by p^k.
    # This is a bit tricky to do efficiently without duplicates in a set.
    # Better: recursive generation.
    
    def gen_smooths(idx, current):
        if idx == len(primes):
            if current <= max_val:
                smooths.add(current)
            return
        p = primes[idx]
        val = current
        while val <= max_val:
            gen_smooths(idx + 1, val)
            val *= p
            
    gen_smooths(0, 1)
    
    smooth_list = list(smooths)
    # Map value to ID
    val_to_id = {v: i for i, v in enumerate(smooth_list)}
    K = len(smooth_list)
    
    # 2. Compute IDs for all x in [1, max_val]
    # ids[x] will be the ID of g(x)
    ids = np.zeros(max_val + 1, dtype=np.int32)
    
    for x in range(1, max_val + 1):
        g = x
        for p in primes:
            while g % p == 0:
                g //= p
        # g is the part with primes >= 24? No.
        # We want the part composed of primes STRICTLY BELOW 24.
        # So we should extract that part.
        # Let's re-calculate.
        pass # The above loop removes the small primes, leaving the large part.
        # We want the SMALL part.
        
    # Correction:
    # g(x) is the product of p^v_p(x) for p < 24.
    # So we should multiply, not divide.
    
    # Let's re-do the ID computation efficiently.
    # Instead of looping x and then primes, we can use a sieve-like approach.
    # But we just need the ID.
    # We can compute g(x) for each x.
    # Since max_val is 1e6, and we have 9 primes, 9e6 operations is fine.
    
    for x in range(1, max_val + 1):
        temp = x
        g = 1
        for p in primes:
            while temp % p == 0:
                g *= p
                temp //= p
        ids[x] = val_to_id[g]
        
    # 3. Sliding Window
    # We need to count distinct IDs in window of size 24.
    counts = [0] * K
    mask = 0
    
    # Initialize window for n=1000: indices 1001 to 1024
    for i in range(1001, 1025):
        id_val = ids[i]
        counts[id_val] += 1
        mask |= (1 << id_val)
        
    min_distinct = mask.bit_count()
    best_n = 1000
    
    # Slide
    # n goes from 1000 to 999999
    # At end of iteration for n, we update for n+1.
    # Remove ids[n+1], Add ids[n+25]
    for n in range(1000, max_n):
        # Check current n
        # (Already checked in init for 1000, so start update for next)
        
        # Prepare for n+1
        out_idx = n + 1
        in_idx = n + 25
        
        out_id = ids[out_idx]
        in_id = ids[in_idx]
        
        counts[out_id] -= 1
        if counts[out_id] == 0:
            mask &= ~(1 << out_id)
            
        counts[in_id] += 1
        if counts[in_id] == 1:
            mask |= (1 << in_id)
            
        new_distinct = mask.bit_count()
        if new_distinct < min_distinct:
            min_distinct = new_distinct
            best_n = n + 1
            
        # Early stop?
        if min_distinct == 1:
            # Theoretically possible?
            # If we find 1, it's the absolute minimum.
            return best_n

    return best_n
