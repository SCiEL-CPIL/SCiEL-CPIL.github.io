"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    TOTAL = 123457
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def generate_smooth_numbers(limit, primes):
        """Generate all numbers <= limit whose prime factors are all in primes."""
        smooth = set()
        smooth.add(1)
        
        # Use a min-heap approach or iterative generation
        # Better: use a sorted list and multiply
        nums = [1]
        result = set([1])
        
        # For each prime, multiply existing numbers
        for p in primes:
            new_nums = []
            for n in list(result):
                val = n * p
                while val <= limit:
                    if val not in result:
                        result.add(val)
                        new_nums.append(val)
                    val *= p
            # Don't add new_nums back to result for this prime iteration to avoid issues
            # Actually we need to be more careful
        
        # Better approach: use a heap
        import heapq
        if not primes:
            return {1}
        
        smooth_set = {1}
        heap = [(1, 0)]  # (value, index into primes)
        
        # Actually, let's use a different approach:
        # Generate all combinations
        smooth_set = set()
        
        def gen(idx, current):
            if current > limit:
                return
            smooth_set.add(current)
            if idx >= len(primes):
                return
            for k in range(idx, len(primes)):
                gen(k, current * primes[k])
                if current * primes[k] > limit:
                    break
        
        gen(0, 1)
        return smooth_set
    
    def gen_smooth(limit, primes):
        """Generate all numbers <= limit with prime factors from primes list."""
        smooth = set([1])
        # Iteratively multiply by each prime
        for p in primes:
            current = list(smooth)
            for n in current:
                val = n * p
                while val <= limit:
                    if val not in smooth:
                        smooth.add(val)
                    val *= p
        return smooth
    
    def find_pair_for_B(B):
        """Check if there exist a,b with a+b=TOTAL, both B-smooth."""
        # Get all primes <= B
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                  53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        primes = [p for p in primes if p <= B]
        
        if not primes:
            return None
        
        smooth = gen_smooth(TOTAL, primes)
        
        # Check for complementary pair
        for a in smooth:
            b = TOTAL - a
            if b >= 1 and b in smooth:
                return (a, b)
        
        return None
    
    # First, let's try small values of B directly
    # The primes to consider as B
    candidate_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                        53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    # Try each B in increasing order
    best_result = (1, TOTAL - 1)  # fallback: P(1)=1, P(123456) is large
    best_B = float('inf')
    
    for B in candidate_primes:
        if time.time() > DEADLINE - 5:
            break
        
        result = find_pair_for_B(B)
        if result is not None:
            best_result = result
            best_B = B
            break
    
    # If no small B worked, try a direct search approach
    if best_B == float('inf'):
        # Try to find any good pair by checking small a values
        # P(1) = 1, so a=1 gives max(1, P(123456))
        # Let's compute P for some candidates
        def largest_prime_factor(n):
            if n <= 1:
                return 1
            lp = 1
            d = 2
            while d * d <= n:
                while n % d == 0:
                    lp = d
                    n //= d
                d += 1
            if n > 1:
                lp = n
            return lp
        
        # Try a few small a values
        best_max = largest_prime_factor(TOTAL - 1)
        best_pair = (1, TOTAL - 1)
        
        for a in range(2, 200):
            if time.time() > DEADLINE - 2:
                break
            b = TOTAL - a
            pa = largest_prime_factor(a)
            pb = largest_prime_factor(b)
            m = max(pa, pb)
            if m < best_max:
                best_max = m
                best_pair = (a, b)
        
        return list(best_pair)
    
    return list(best_result)

# EVOLVE-BLOCK-END
