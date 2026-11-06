import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    target = 123457
    
    # Generate primes up to a reasonable limit
    # We can stop if we find a solution
    # Let's generate primes up to 1000 first
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, val in enumerate(is_prime) if val]
    
    primes = sieve_primes(1000)
    
    # Function to check if a number is B-smooth (primes in 'primes_list')
    def is_smooth(n, primes_list):
        if n == 1:
            return True
        for p in primes_list:
            while n % p == 0:
                n //= p
            if n == 1:
                return True
            if p * p > n:
                break
        return n == 1

    # Iterate over increasing sets of primes
    # k is the index in primes list, so we use primes[0:k+1]
    for k in range(len(primes)):
        if time.time() > deadline - 5:
            # Fallback to a simple split if we run out of time
            # This should not happen for small k
            return [target // 2, target - target // 2]
            
        current_primes = primes[:k+1]
        
        # Generate all smooth numbers <= target-1
        smooth_nums = []
        for i in range(1, target):
            if is_smooth(i, current_primes):
                smooth_nums.append(i)
        
        # Convert to set for O(1) lookup
        smooth_set = set(smooth_nums)
        
        # Check for pair
        for a in smooth_nums:
            b = target - a
            if b in smooth_set:
                # Found a pair with max prime factor <= current_primes[-1]
                # Since we iterate k in increasing order, this is the minimal max prime factor
                return [a, b]
                
    # If no solution found with primes up to 1000, fall back to full sieve
    # This is unlikely but safe
    max_n = target - 1
    lpf = [0] * (max_n + 1)
    lpf[1] = 1
    for i in range(2, max_n + 1):
        if lpf[i] == 0:
            for j in range(i, max_n + 1, i):
                lpf[j] = i
                
    best_max = float('inf')
    best_pair = (1, target - 1)
    for a in range(1, target):
        b = target - a
        cur_max = max(lpf[a], lpf[b])
        if cur_max < best_max:
            best_max = cur_max
            best_pair = (a, b)
            
    return [best_pair[0], best_pair[1]]
