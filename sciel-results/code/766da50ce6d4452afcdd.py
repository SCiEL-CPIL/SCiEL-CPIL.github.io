import time
import math

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin for 90s limit
    
    N = 1000000
    lower = 10000
    
    # Generate primes up to a reasonable limit
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    # We'll try increasing smoothness bounds
    # For each B, generate all B-smooth numbers up to N+1
    # and check for consecutive pairs in [lower, N]
    
    # First, let's get primes up to 200 to have a reasonable set
    primes = sieve_primes(200)
    
    best_lpf = float('inf')
    best_n = lower
    
    # For each subset of primes (increasing in size), generate smooth numbers
    # Start with small prime sets
    
    def generate_smooth_numbers(prime_list, max_val):
        """Generate all numbers <= max_val that are products of primes in prime_list."""
        if not prime_list:
            return {1}
        
        result = {1}
        for p in prime_list:
            new_entries = set()
            for x in result:
                val = x
                while val * p <= max_val:
                    val *= p
                    if val not in result:
                        new_entries.add(val)
            result |= new_entries
        return result
    
    # Try increasing sets of primes
    # Start with the smallest primes
    for num_primes in range(1, len(primes) + 1):
        if time.time() - start_time > deadline:
            break
        
        current_primes = primes[:num_primes]
        B = current_primes[-1]  # The largest prime in our set
        
        # If B is already larger than our best, we can stop
        if B >= best_lpf:
            break
        
        # Generate all B-smooth numbers up to N+1
        smooth_nums = generate_smooth_numbers(current_primes, N + 1)
        
        # Convert to sorted list and check for consecutive pairs
        sorted_smooth = sorted(smooth_nums)
        
        # Use a set for O(1) lookup
        smooth_set = set(sorted_smooth)
        
        # Check for consecutive pairs in range [lower, N]
        for n in sorted_smooth:
            if n < lower or n > N:
                continue
            if (n + 1) in smooth_set:
                # Both n and n+1 are B-smooth
                # LPF of n(n+1) is at most B
                if B < best_lpf:
                    best_lpf = B
                    best_n = n
        
        # Check time
        if time.time() - start_time > deadline - 5:
            break
    
    return best_n
