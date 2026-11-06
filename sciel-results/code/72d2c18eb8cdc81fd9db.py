"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    N = 123457
    deadline = time.time() + 80.0  # Safety margin before 90s limit
    
    # Generate primes up to N using simple sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Binary search on the smoothness bound B
    # We want to find the minimum B such that there exist a, b with a+b=N, 
    # and both a and b are B-smooth (all prime factors <= B)
    
    # First, let's establish a high bound and low bound
    # Low bound: 2 (only powers of 2, but N is odd so one must be 1... wait)
    # Actually, let's just do a linear scan of primes as potential B values, 
    # but we can be smarter. Let's use binary search on the index in primes list.
    
    def is_b_smooth(n, b):
        """Check if n is B-smooth using primes up to B"""
        if n == 1:
            return True
        for p in primes:
            if p > b:
                break
            if p * p > n:
                break
            while n % p == 0:
                n //= p
        return n == 1
    
    def check_smooth_pair_exists(b):
        """Check if there exists a pair (a, b) with a+b=N where both are b-smooth"""
        if time.time() > deadline:
            return False, None
        
        # Generate all b-smooth numbers up to N
        smooth_numbers = set()
        
        def gen_smooth(current, prime_idx):
            if time.time() > deadline:
                return
            if current > N:
                return
            smooth_numbers.add(current)
            if prime_idx >= len(primes):
                return
            p = primes[prime_idx]
            if p > b:
                return
            # Multiply by p repeatedly
            val = current
            while val <= N:
                gen_smooth(val, prime_idx + 1)
                if val > N // p:
                    break
                val *= p
        
        gen_smooth(1, 0)
        
        # Check for complementary pair
        for a in smooth_numbers:
            if a >= N:
                continue
            b_comp = N - a
            if b_comp in smooth_numbers:
                return True, (a, b_comp)
        
        return False, None
    
    # Binary search on the prime index
    # The answer must be one of the primes (or 1)
    # Let's try to find the minimum prime p such that a pair exists with max prime factor <= p
    
    # We know the answer is at most the largest prime factor of any number in a valid pair
    # Let's do a linear scan of primes, but stop early if we find a solution
    # Actually, binary search on the sorted list of primes
    
    lo = 0
    hi = len(primes) - 1
    best_solution = None
    best_b = None
    
    # First check if a solution exists with the largest prime
    # Then binary search for the minimum
    
    # Actually, let's just iterate through primes in order and check each one
    # Since the number of primes up to 123457 is about 11500, and generating smooth numbers
    # might be expensive, let's be careful.
    
    # Alternative: Just compute LPF directly but with a faster method
    # The original method was O(N log log N) which is fast enough.
    # Let's stick with the original approach but optimize it slightly.
    
    # Actually, let's try the binary search approach as suggested in the directive.
    # But generating all B-smooth numbers can be expensive for large B.
    
    # Let's try a hybrid: use the LPF precomputation but with a segmented sieve or 
    # a more efficient method.
    
    # For now, let's use the original LPF approach but with a time guard.
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    for i in range(2, N + 1):
        if time.time() > deadline:
            break
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_max_p = float('inf')
    best_a = 1
    
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_max_p:
            best_max_p = val
            best_a = a
            
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
