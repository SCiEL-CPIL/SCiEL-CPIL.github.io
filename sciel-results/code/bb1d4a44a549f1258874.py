import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 105  # 15s margin
    
    # 1. Sieve of Eratosthenes for primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Fast phi function
    def phi(n):
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result

    best_k = 1
    best_s = 1
    max_n = 10**12
    
    # 3. Search Strategy
    # We will sample random starts and also try to extend from promising areas.
    # Given the "Knowledge" hint, random sampling in large ranges is effective.
    
    # We can also try a local search: if we find a run of length k at s, 
    # check s-1 and s+k to see if we can extend.
    
    # Let's do a mix of random sampling and local refinement.
    # Number of samples depends on time.
    
    # Helper to check run length starting at s
    def get_run_length(s):
        if s + 200 > max_n:
            return 0
        seen = set()
        k = 0
        for i in range(200):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    # We will run random samples until deadline is near.
    # To ensure we don't miss small ranges, let's do a quick scan of 1..1000 first.
    for s in range(1, 1000):
        if time.time() > deadline:
            break
        k = get_run_length(s)
        if k > best_k:
            best_k = k
            best_s = s

    # Now random sampling in [1, 10^12]
    # We can use a fixed seed for reproducibility if needed, but random is fine.
    random.seed(42)
    
    # Estimate phi cost: ~100-200 operations per number on average? 
    # Let's just run as many as we can.
    
    while time.time() < deadline:
        # Pick a random start
        # Bias towards larger numbers? Uniform is fine.
        s = random.randint(1, max_n - 200)
        
        # Check if we are close to a previously found best? 
        # No, just check the run.
        k = get_run_length(s)
        
        if k > best_k:
            best_k = k
            best_s = s
            
            # Local refinement: try to extend left and right
            # Extend left
            new_s = best_s
            while new_s > 1:
                # Check if phi(new_s - 1) is distinct from phi(new_s)...phi(new_s + best_k - 1)
                # This requires recomputing the set or checking carefully.
                # Let's just re-evaluate the run starting at new_s - 1
                # If get_run_length(new_s - 1) > best_k, update.
                # This is expensive but we only do it when we find a new best.
                k_left = get_run_length(new_s - 1)
                if k_left > best_k:
                    best_k = k_left
                    best_s = new_s - 1
                    new_s = best_s
                else:
                    break
            
            # Extend right
            new_s = best_s
            while new_s + best_k < max_n:
                k_right = get_run_length(new_s) # This starts at new_s, but we know length is best_k. 
                # Actually, get_run_length(new_s) will just return best_k or more if it extends.
                # But we need to check if it extends.
                # If get_run_length(new_s) > best_k, it means it extended to the right? 
                # No, get_run_length(s) returns the max length starting at s.
                # If we are at best_s with length best_k, and we check best_s+1, 
                # the length might be best_k - 1 (if the first element was unique to the start) or more.
                # Actually, if we have a run [s, s+k-1], the run starting at s+1 is a subset.
                # So get_run_length(s+1) <= k. It cannot extend to the right from s+1 
                # to exceed k unless the "collision" that stopped the run at s+k was 
                # caused by an element in [s, s+k-1]. 
                # If phi(s) == phi(s+k), then starting at s+1, we don't have phi(s), 
                # so maybe we can go further?
                # Yes! The collision might be between the start and the end.
                # So checking s+1 is valuable.
                
                k_next = get_run_length(new_s + 1)
                if k_next > best_k:
                    best_k = k_next
                    best_s = new_s + 1
                    new_s = best_s
                else:
                    # It's possible that s+1 doesn't extend, but s+2 does? Unlikely to jump.
                    # But let's check s+1 only.
                    break

    return {'start': best_s, 'length': best_k}
