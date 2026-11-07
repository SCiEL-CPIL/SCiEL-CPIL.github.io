import time
from math import isqrt

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # We need to search for M. 
    # Based on previous scores, M is likely small (score 22.5 implies max_val - 210 = 22.5? No, score is seed/value. 
    # If score is 22.5 and seed is 1, value is 1/22.5? No, "search score is seed/value... 1.0 equals seed".
    # Usually "value" in these tasks is the metric to minimize. 
    # "raw value is max(selected) - 210".
    # If score is 22.5, and seed is 1, then 1/value = 22.5 => value = 1/22.5 = 0.044? That doesn't make sense for an integer difference.
    # Re-reading: "The evaluator measures the raw value (lower is better). The search score is seed/value... 1.0 equals the seed program".
    # This implies if the seed program had value V_seed, its score is 1.0.
    # If my program has value V_me, score = V_seed / V_me.
    # If score is 22.5, then V_seed / V_me = 22.5.
    # This means V_me = V_seed / 22.5.
    # This implies V_me is much smaller than V_seed.
    # Or, maybe the "score" printed in the prompt is the *value* itself?
    # "CURRENT PROGRAM (score 22.5000)"
    # If the raw value is max(selected) - 210, and the score is 22.5, maybe the raw value IS 22.5?
    # But max(selected) and 210 are integers, so max(selected) - 210 must be an integer.
    # So 22.5 cannot be the raw value.
    # Perhaps the "score" in the header is the *evaluation score*?
    # If score = 22.5, and it's an improvement, it's good.
    # Let's assume the goal is to minimize `max(selected) - 210`.
    # Let's aim for the smallest possible `max(selected)`.
    
    max_n_limit = 20000
    # Precompute SPF
    spf = list(range(max_n_limit + 1))
    for i in range(2, int(max_n_limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_n_limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_sf(n):
        if n <= 1:
            return 1
        result = 1
        x = n
        last_p = 0
        while x > 1:
            p = spf[x]
            if p == last_p:
                x //= p
                continue
            cnt = 0
            while x % p == 0:
                x //= p
                cnt ^= 1
            if cnt:
                result *= p
            last_p = p
        return result

    # Precompute square-free parts for range 211 to max_n_limit
    sf_arr = [0] * (max_n_limit + 1)
    sf_arr[1] = 1
    for i in range(2, max_n_limit + 1):
        sf_arr[i] = get_sf(i)
    
    target_sf = 210
    
    # Collect primes up to max_n_limit
    primes = []
    for i in range(2, max_n_limit + 1):
        if spf[i] == i:
            primes.append(i)
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def sf_to_bits(sf_val):
        bits = 0
        x = sf_val
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                bits |= (1 << prime_to_idx[p])
                x //= p
        if x > 1:
            if x in prime_to_idx:
                bits |= (1 << prime_to_idx[x])
        return bits
    
    target_bits = sf_to_bits(target_sf)
    
    # Precompute bit vectors for all numbers in range
    # We only care about numbers 211 and above
    bits_arr = [0] * (max_n_limit + 1)
    for i in range(211, max_n_limit + 1):
        bits_arr[i] = sf_to_bits(sf_arr[i])

    def find_subset(M):
        """
        Find subset of {211, ..., M} whose XOR of bits equals target_bits.
        Returns list of indices (actual numbers) or None.
        """
        if M < 211:
            return None
        
        # Extract relevant vectors
        # We process them in order 211 to M
        basis = {} # pivot_bit -> (vector, frozenset_of_indices)
        
        # Optimization: We can just iterate 211 to M
        for i in range(211, M + 1):
            v = bits_arr[i]
            if v == 0:
                continue
            
            cur = v
            used = {i}
            
            # Reduce cur by existing basis
            # Find highest set bit
            hb = cur.bit_length() - 1
            while hb >= 0 and hb in basis:
                bv, bused = basis[hb]
                cur ^= bv
                used = used.symmetric_difference(bused)
                hb = cur.bit_length() - 1 if cur != 0 else -1
            
            if cur == 0:
                continue
            
            hb = cur.bit_length() - 1
            basis[hb] = (cur, frozenset(used))
        
        # Check if target is in span
        if target_bits == 0:
             # Need non-empty subset XORing to 0. 
             # If any v was 0, we could pick it. But sf(n) != 0 for n>1.
             # So we need a dependent set.
             # For now, assume target != 0.
             return None

        cur = target_bits
        used = set()
        hb = cur.bit_length() - 1
        while hb >= 0 and hb in basis:
            bv, bused = basis[hb]
            cur ^= bv
            used = used.symmetric_difference(set(bused))
            hb = cur.bit_length() - 1 if cur != 0 else -1
        
        if cur != 0:
            return None
        
        if not used:
            return None
            
        # Check size constraint
        if len(used) > 100:
            # This specific solution is invalid due to size.
            # There might be another solution in the affine space.
            # However, finding a smaller weight solution is hard.
            # We return None to force search for larger M? 
            # No, if a solution exists, it might be that all solutions have weight > 100.
            # But for small M, weights are usually small.
            # If we return None, we might miss a valid M if the *found* solution is too big but another is small.
            # Given the random nature, the first found is likely near minimal weight.
            # Let's return the solution anyway and let the caller check? 
            # The task says "1..100 distinct integers". If we return >100, it's invalid.
            # So we must return None if the found solution is > 100.
            return None
        
        return sorted(list(used))

    # Binary search for minimal M
    lo = 211
    hi = max_n_limit
    best_solution = None
    best_M = max_n_limit + 1
    
    # First check if hi works
    if time.time() < deadline - 5.0:
        res_hi = find_subset(hi)
        if res_hi is None:
            # If even max_n_limit doesn't work, we have a problem.
            # But it should work.
            return [224, 240] # Fallback
            
    while lo < hi:
        if time.time() > deadline - 10.0:
            break
        mid = (lo + hi) // 2
        res_mid = find_subset(mid)
        if res_mid is not None:
            best_solution = res_mid
            best_M = mid
            hi = mid
        else:
            lo = mid + 1
            
    # Check lo (which equals hi at end of loop, or lo was updated)
    if time.time() < deadline - 5.0:
        res_lo = find_subset(lo)
        if res_lo is not None:
            best_solution = res_lo
            best_M = lo

    if best_solution is None:
        # Fallback to linear scan for small M if binary search failed or returned None
        # This can happen if the "size > 100" check triggered for all mid points 
        # but a small M actually works? Unlikely.
        # Or if the span check failed due to some bug.
        for M in range(211, 500):
             if time.time() > deadline - 5.0:
                 break
             res = find_subset(M)
             if res is not None:
                 best_solution = res
                 break

    if best_solution is None:
        best_solution = [224, 240]
        
    # Final verification
    prod = 210
    for x in best_solution:
        prod *= x
    if not _is_perfect_square(prod):
        # Should not happen
        best_solution = [224, 240]
        
    result = [int(x) for x in best_solution]
    return result
