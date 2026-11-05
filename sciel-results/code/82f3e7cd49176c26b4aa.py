import time
import math
import random

def construct():
    start = time.time()
    DEADLINE = start + 80  # Safety margin

    # Precompute primes up to 256
    _is_prime_arr = [True] * 257
    _is_prime_arr[0] = False
    _is_prime_arr[1] = False
    for i in range(2, int(256**0.5) + 1):
        if _is_prime_arr[i]:
            for j in range(i*i, 257, i):
                _is_prime_arr[j] = False

    primes = [i for i in range(2, 257) if _is_prime_arr[i]]
    
    # We want to find A, B such that all 9 sums are distinct primes.
    # Structure:
    # a1 = x
    # b1 = p1 - x
    # b2 = p2 - x
    # b3 = p3 - x
    # a2 = p4 - p1 + x
    # a3 = p6 - p1 + x
    # p5 = p4 - p1 + p2
    # p7 = p6 - p1 + p2
    # p8 = p4 - p1 + p3
    # p9 = p6 - p1 + p3
    
    # Constraints:
    # 1. p1, p2, p3, p4, p5, p6, p7, p8, p9 must be primes.
    # 2. All 9 sums must be distinct. (Since they are primes, if distinct values, they are distinct primes).
    #    Note: If p5 == p1, they are not distinct. So we need p1..p9 to be distinct values.
    # 3. a1, a2, a3, b1, b2, b3 must be in [1, 128] and distinct.
    #    a1 = x
    #    a2 = x + (p4 - p1)
    #    a3 = x + (p6 - p1)
    #    b1 = p1 - x
    #    b2 = p2 - x
    #    b3 = p3 - x
    
    # Let's iterate over p1, p2, p3, p4, p6.
    # To make it fast, we can fix p1, p2, p3 and iterate p4, p6.
    # Or just random sampling with a fixed seed for determinism and speed.
    
    rng = random.Random(42)
    
    # Try a large number of random combinations
    # 5 primes chosen from ~50 primes.
    # We need p5, p7, p8, p9 to be primes.
    # Probability is roughly (1/log(p))^4. For p~100, log~4.6. (1/4.6)^4 ~ 1/450.
    # So 1 in 450 random 5-tuples works.
    # 1000 tries should be enough.
    
    for _ in range(5000):
        if time.time() >= DEADLINE:
            break
        
        p1, p2, p3, p4, p6 = rng.sample(primes, 5)
        
        p5 = p4 - p1 + p2
        p7 = p6 - p1 + p2
        p8 = p4 - p1 + p3
        p9 = p6 - p1 + p3
        
        # Check if derived primes are valid
        if p5 < 2 or p5 > 256 or p7 < 2 or p7 > 256 or p8 < 2 or p8 > 256 or p9 < 2 or p9 > 256:
            continue
        if not _is_prime_arr[p5] or not _is_prime_arr[p7] or not _is_prime_arr[p8] or not _is_prime_arr[p9]:
            continue
            
        # Check distinctness of the 9 primes
        ps = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
        if len(set(ps)) != 9:
            continue
            
        # Now find x
        # x must satisfy:
        # 1 <= x <= 128
        # 1 <= p4 - p1 + x <= 128  =>  1 - (p4 - p1) <= x <= 128 - (p4 - p1)
        # 1 <= p6 - p1 + x <= 128  =>  1 - (p6 - p1) <= x <= 128 - (p6 - p1)
        # 1 <= p1 - x <= 128       =>  p1 - 128 <= x <= p1 - 1
        # 1 <= p2 - x <= 128       =>  p2 - 128 <= x <= p2 - 1
        # 1 <= p3 - x <= 128       =>  p3 - 128 <= x <= p3 - 1
        
        low = max(1, 
                  1 - (p4 - p1), 
                  1 - (p6 - p1), 
                  p1 - 128, 
                  p2 - 128, 
                  p3 - 128)
                  
        high = min(128, 
                   128 - (p4 - p1), 
                   128 - (p6 - p1), 
                   p1 - 1, 
                   p2 - 1, 
                   p3 - 1)
                   
        if low > high:
            continue
            
        # Try to find x in [low, high] such that a1, a2, a3, b1, b2, b3 are distinct
        # a1 = x
        # a2 = x + p4 - p1
        # a3 = x + p6 - p1
        # b1 = p1 - x
        # b2 = p2 - x
        # b3 = p3 - x
        
        # Since the range is small, we can scan
        found = False
        for x in range(low, high + 1):
            a1 = x
            a2 = x + p4 - p1
            a3 = x + p6 - p1
            b1 = p1 - x
            b2 = p2 - x
            b3 = p3 - x
            
            vals = [a1, a2, a3, b1, b2, b3]
            if len(set(vals)) == 6:
                # Verify sums are correct and in range
                A = [a1, a2, a3]
                B = [b1, b2, b3]
                
                # Final check
                sums = set()
                valid = True
                for a in A:
                    for b in B:
                        s = a + b
                        if s < 1 or s > 256:
                            valid = False
                            break
                        sums.add(s)
                if not valid:
                    continue
                    
                # Check pairwise coprime
                # Since we ensured they are distinct primes, they are pairwise coprime.
                # But just in case, verify.
                sum_list = list(sums)
                ok = True
                for i in range(len(sum_list)):
                    for j in range(i+1, len(sum_list)):
                        if math.gcd(sum_list[i], sum_list[j]) != 1:
                            ok = False
                            break
                    if not ok:
                        break
                            
                if ok and len(sums) == 9:
                    return [A, B]
                    
    # Fallback: Return a known valid solution or best effort
    # If we didn't find 9, we can try to find a valid one with fewer sums.
    # But 9 is the target.
    # Let's try a few more random starts if we have time.
    
    # As a last resort, return a valid pair with some coprime sums.
    # Example: A=[1, 3, 5], B=[2, 4, 6]
    # Sums: 3, 5, 7, 5, 7, 9, 7, 9, 11.
    # Distinct: 3, 5, 7, 9, 11.
    # 9 is not coprime with 3.
    # Let's just return [1, 2, 3], [4, 5, 6] as a valid (but low score) fallback?
    # No, we should try to find something better.
    
    # Simple search for any valid pair
    for _ in range(10000):
        if time.time() >= DEADLINE:
            break
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        
        sums = set()
        for a in A:
            for b in B:
                sums.add(a+b)
                
        sum_list = list(sums)
        n = len(sum_list)
        if n == 0:
            continue
            
        conflict = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(sum_list[i], sum_list[j]) != 1:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                    
        max_count = 0
        for mask in range(1 << n):
            is_independent = True
            for i in range(n):
                if mask & (1 << i):
                    if mask & conflict[i]:
                        is_independent = False
                        break
            if is_independent:
                count = bin(mask).count('1')
                if count > max_count:
                    max_count = count
                    
        if max_count >= 5:
            return [A, B]
            
    return [[1, 2, 3], [4, 5, 6]]
