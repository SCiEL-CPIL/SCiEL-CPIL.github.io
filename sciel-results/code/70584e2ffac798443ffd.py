def construct():
    import time
    from fractions import Fraction
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_denoms = []
    
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # The first denominator d1 must satisfy 1/d1 < 5/31 => d1 > 6.2
    # So d1 starts at 7.
    # We can bound d1. If d1 is too large, the remaining fraction 5/31 - 1/d1 
    # will likely require larger denominators.
    # Let's iterate d1 from 7 to 200.
    
    for d1 in range(7, 200):
        if time.time() > deadline:
            break
            
        r = target - Fraction(1, d1)
        if r <= 0:
            continue
            
        used = {d1}
        denoms = [d1]
        
        # Greedy completion
        valid = True
        while r > 0:
            # Find smallest k not in used such that 1/k <= r
            # k >= ceil(1/r) = ceil(r.denominator / r.numerator)
            k_min = (r.denominator + r.numerator - 1) // r.numerator
            
            # Find the smallest k >= k_min not in used
            k = k_min
            while k in used:
                k += 1
                
            # Check if 1/k <= r
            # Since k >= ceil(1/r), 1/k <= 1/ceil(1/r) <= r is not always true?
            # Wait. If r = 3/7, ceil(7/3) = 3. 1/3 <= 3/7? 7 <= 9. Yes.
            # If r = 2/5, ceil(5/2) = 3. 1/3 <= 2/5? 5 <= 6. Yes.
            # Generally, let k = ceil(1/r). Then k >= 1/r => 1/k <= r.
            # So the condition 1/k <= r is automatically satisfied for k = ceil(1/r).
            # And for any k > k_min, 1/k < 1/k_min <= r.
            
            denoms.append(k)
            used.add(k)
            r -= Fraction(1, k)
            
            # Optimization: if r becomes 0, we are done
            # If r is very small, we might hit a long chain, but greedy is fast.
            
        if denoms:
            current_max = max(denoms)
            if current_max < best_max:
                best_max = current_max
                best_denoms = denoms[:]
                
    return best_denoms
