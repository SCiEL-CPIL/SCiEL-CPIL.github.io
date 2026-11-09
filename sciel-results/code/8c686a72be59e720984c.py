import math
import time

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    max_val = P['maximum_window']
    max_size = P['maximum_size']
    
    target_sig = frozenset({2, 3, 5, 7})
    
    # Primes up to 47 are "small" enough that 210*p <= 10210
    # 210 * 47 = 9870
    # 210 * 53 = 11130 > 10210
    # So any number with a prime factor > 47 cannot be part of a 2-element solution 
    # where the other number is <= 10210, UNLESS the other number also has that factor,
    # which would make it >= 210 * p > 10210.
    # Wait, if a has factor p, b must have factor p.
    # If p > 47, b >= 210 * p > 10210.
    # So for 2-element solutions, all prime factors of a and b must be <= 47.
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def get_parity_vector_fast(x):
        """Returns frozenset of primes with odd exponent. Returns None if x has a prime factor > 47."""
        sig = set()
        temp = x
        for p in small_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    sig.add(p)
        if temp > 1:
            # If temp > 1, it must be a prime (since we divided out all small primes)
            # If temp > 47, it's a "large" prime
            if temp > 47:
                return None
            else:
                sig.add(temp)
        return frozenset(sig)

    best_answer = None
    best_value = float('inf')
    
    # 1. Check single element solutions
    # x = 210 * k^2
    # x > 210 => k >= 1
    # x <= 10210 => 210 * k^2 <= 10210 => k^2 <= 48.6 => k <= 6
    for k in range(1, 7):
        x = 210 * k * k
        if 210 < x <= max_val:
            val = x - 210
            if val < best_value:
                best_value = val
                best_answer = [x]
                
    # 2. Check 2-element solutions
    # We need sig(a) ^ sig(b) = target_sig
    # Both a and b must have only small prime factors.
    
    # Map signature to the smallest number with that signature
    sig_to_min = {}
    
    for x in range(211, max_val + 1):
        if time.time() > deadline:
            break
        sig = get_parity_vector_fast(x)
        if sig is not None:
            if sig not in sig_to_min or x < sig_to_min[sig]:
                sig_to_min[sig] = x
    
    # Iterate through signatures to find pairs
    # To avoid duplicate checks (a,b) and (b,a), we can just check all
    # and take the max.
    for sig, min_num in sig_to_min.items():
        needed_sig = sig ^ target_sig
        if needed_sig in sig_to_min:
            partner = sig_to_min[needed_sig]
            if min_num != partner:
                max_elem = max(min_num, partner)
                val = max_elem - 210
                if val < best_value:
                    # Verify
                    prod = 210 * min_num * partner
                    sqrt_val = math.isqrt(prod)
                    if sqrt_val * sqrt_val == prod:
                        best_value = val
                        best_answer = [min_num, partner]
                        
    # 3. Check 3-element solutions?
    # If 2-element didn't beat single, maybe 3-element can?
    # Min single is 630.
    # Can we get max < 840 with 3 elements?
    # sig(a)^sig(b)^sig(c) = target
    # We can try to find small a, b, c.
    # Given the time limit and that 2-element search is fast, let's try a limited 3-element search
    # if best_value is still high.
    
    if best_value > 100: # Heuristic: if we haven't found something very good, try 3 elements
        # We want to minimize max(a,b,c).
        # Let's iterate max_elem from 211 upwards? No, that's slow.
        # Let's iterate pairs (a,b) with small values and check if c exists.
        # To keep it fast, restrict a, b to numbers with small prime factors.
        # And restrict a, b to be < current best_value + 210.
        
        limit = 210 + best_value
        if limit > max_val: limit = max_val
        
        # Get list of candidates (num, sig)
        candidates = []
        for x in range(211, limit + 1):
            sig = get_parity_vector_fast(x)
            if sig is not None:
                candidates.append((x, sig))
                
        # Sort by value
        candidates.sort()
        
        # Try pairs
        # We can break early if a >= best_value + 210
        for i in range(len(candidates)):
            a, sig_a = candidates[i]
            if a - 210 >= best_value:
                break
            for j in range(i + 1, len(candidates)):
                b, sig_b = candidates[j]
                if b - 210 >= best_value:
                    break
                sig_c = sig_a ^ sig_b ^ target_sig
                if sig_c in sig_to_min:
                    c = sig_to_min[sig_c]
                    if c != a and c != b and c <= limit:
                        max_elem = max(a, b, c)
                        val = max_elem - 210
                        if val < best_value:
                            prod = 210 * a * b * c
                            sqrt_val = math.isqrt(prod)
                            if sqrt_val * sqrt_val == prod:
                                best_value = val
                                best_answer = [a, b, c]
                                # Update limit? No, just continue to see if we can go lower
                                # Actually, if we found a better one, we can tighten the break condition
                                # But the loop structure makes it hard to dynamically adjust break.
                                # We can just let it run, it's bounded by limit.

    if best_answer is not None:
        return [int(x) for x in best_answer]
    return [int(840)]
