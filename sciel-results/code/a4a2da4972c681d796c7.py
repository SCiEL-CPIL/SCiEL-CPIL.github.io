import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    target = 123457
    # Fallback solution
    best_a, best_b = 1, target - 1
    
    # Generate primes up to a reasonable limit
    limit_primes = 10000
    is_prime = [True] * (limit_primes + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit_primes)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_primes + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Set of smooth numbers generated so far
    smooth_nums = {1}
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Generate new smooth numbers involving prime p
        new_nums = set()
        old_smooth = list(smooth_nums)
        for x in old_smooth:
            val = x
            while val * p <= target:
                val *= p
                new_nums.add(val)
        
        # Check if any new number has its complement in smooth_nums
        # This is much faster than checking all smooth_nums
        for a in new_nums:
            b = target - a
            if b in smooth_nums:
                return [a, b]
            # Also check the reverse: if b was added earlier and a is new
            # Actually we should also check if a in smooth_nums and b in new_nums
            # But since we're iterating new_nums, let's also check complements
            if a in smooth_nums and b in new_nums:
                # This case is handled when we iterate to b
                pass
        
        # Add new numbers to smooth set
        smooth_nums.update(new_nums)
        
        # Also need to check: for a in smooth_nums (old), if target-a is in new_nums
        # We already checked new_nums against old smooth_nums above
        # But we should also check old smooth_nums against new_nums
        # Actually, if a is old and b is new, then when we checked b (new) against smooth_nums (old), we'd find a
        # So the check above is sufficient
        
    # If loop finishes without finding a solution (or timeout), return fallback
    return [best_a, best_b]
