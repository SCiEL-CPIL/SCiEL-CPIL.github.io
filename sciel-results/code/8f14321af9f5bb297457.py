import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    def list_to_bits(lst):
        bits = 0
        for x in lst:
            bits |= (1 << x)
        return bits
            
    def bits_to_set(bits):
        return [i for i in range(1, 257) if (bits >> i) & 1]
        
    def can_add(S_bits, x):
        """
        Checks if x can be added to the set represented by S_bits.
        Returns True if valid, False otherwise.
        """
        # Case 1: x is the smallest element (a=x)
        # Need b, c in S such that x < b <= c and x | (b+c)
        # b+c must be a multiple of x, say m.
        # Since b > x and c >= b, b+c >= 2*(x+1) = 2x+2.
        # Max b+c is 256+256=512.
        for m in range(2 * x + 2, 513, x):
            # b must be in [max(x+1, m-256), min(m//2, 256)]
            b_lo = max(x + 1, m - 256)
            b_hi = min(m // 2, 256)
            if b_lo > b_hi:
                continue
            # Check if there exists b in range such that b in S and m-b in S
            # We iterate b. To speed up, we could use bit tricks, but loop is fine for small ranges.
            for b in range(b_lo, b_hi + 1):
                c = m - b
                if c >= b and ((S_bits >> b) & 1) and ((S_bits >> c) & 1):
                    return False
        
        # Case 2: x is the middle element (b=x)
        # Need a, c in S such that a < x <= c and a | (x+c)
        # Iterate over a in S where a < x
        a_mask = S_bits & ((1 << x) - 1)
        while a_mask:
            a = (a_mask & -a_mask).bit_length() - 1
            a_mask &= a_mask - 1
            
            # x+c must be a multiple of a, say k*a.
            # x+c = k*a => c = k*a - x
            # Constraints: c >= x and c <= 256
            # k*a - x >= x => k*a >= 2x => k >= ceil(2x/a)
            # k*a - x <= 256 => k*a <= 256+x => k <= floor((256+x)/a)
            k_lo = math.ceil(2 * x / a)
            k_hi = (256 + x) // a
            for k in range(k_lo, k_hi + 1):
                c = k * a - x
                if c >= x and ((S_bits >> c) & 1):
                    return False
        
        # Case 3: x is the largest element (c=x)
        # Need a, b in S such that a < b <= x and a | (b+x)
        # Iterate over a in S where a < x
        a_mask = S_bits & ((1 << x) - 1)
        while a_mask:
            a = (a_mask & -a_mask).bit_length() - 1
            a_mask &= a_mask - 1
            
            # b+x must be a multiple of a, say k*a.
            # b+x = k*a => b = k*a - x
            # Constraints: a < b <= x
            # k*a - x > a => k*a > x+a => k > (x+a)/a => k >= floor((x+a)/a) + 1
            # k*a - x <= x => k*a <= 2x => k <= floor(2x/a)
            k_lo = (x + a) // a + 1
            k_hi = (2 * x) // a
            for k in range(k_lo, k_hi + 1):
                b = k * a - x
                if a < b <= x and ((S_bits >> b) & 1):
                    return False
        
        return True

    def greedy_fill(S_bits, candidates):
        for x in candidates:
            if (S_bits >> x) & 1:
                continue
            if can_add(S_bits, x):
                S_bits |= (1 << x)
        return S_bits

    best_bits = 0
    best_size = 0
    
    # Try different tail starts.
    # We know 171 is valid. Higher starts are also valid but smaller.
    # Lower starts might be valid but require checking.
    # We'll stick to known valid ones and let the greedy try to add lower elements.
    tail_starts = [171, 175, 180, 185, 190, 195]
    
    for tail_start in tail_starts:
        if time.time() > deadline - 10:
            break
            
        # Verify tail validity quickly (optional, but good for safety)
        # For these starts, we are confident they are valid based on math.
        base_tail = list(range(tail_start, 257))
        base_bits = list_to_bits(base_tail)
        base_size = len(base_tail)
        
        if base_size > best_size:
            best_bits = base_bits
            best_size = base_size
            
        candidates = list(range(1, tail_start))
        
        # Run multiple random restarts for this tail start
        rng = random.Random(tail_start * 1000)
        
        # Number of restarts depends on time left
        # Estimated time per restart: ~1-5 ms (optimistic) to ~50ms (pessimistic)
        # Let's aim for 100-500 restarts if fast.
        
        restarts_done = 0
        while time.time() < deadline - 5:
            order = candidates[:]
            rng.shuffle(order)
            S_bits = greedy_fill(base_bits, order)
            size = bin(S_bits).count('1')
            if size > best_size:
                best_bits = S_bits
                best_size = size
            restarts_done += 1
            
            # If we are getting very few improvements, we might be stuck, 
            # but random restarts help escape local optima of the greedy order.
            # We just keep going until time runs out.
            
            # Safety break if too slow
            if restarts_done > 2000:
                break

    return bits_to_set(best_bits)
