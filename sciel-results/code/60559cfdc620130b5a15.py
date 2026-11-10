import time
import sys

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: 50 seconds
    deadline = time.time() + 50.0
    
    LOW = 10000
    HIGH = 1000000
    
    # Primes to test in increasing order
    # We want to minimize the max prime factor.
    # So we test sets of primes with small max element first.
    
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19],
        [2, 3, 5, 7, 11, 13, 17, 19, 23],
    ]
    
    def generate_smooth(primes, limit):
        """Generate all numbers of the form p1^a * p2^b * ... <= limit"""
        res = set()
        n_primes = len(primes)
        
        # Recursive helper
        def rec(idx, current):
            if current > limit:
                return
            if idx == n_primes:
                res.add(current)
                return
            p = primes[idx]
            val = 1
            while current * val <= limit:
                rec(idx + 1, current * val)
                val *= p
        
        rec(0, 1)
        return res

    def find_best(primes):
        """Find n in [LOW, HIGH] such that n and n+1 are p-smooth for p in primes."""
        if time.time() > deadline:
            return None
            
        # Generate smooth numbers up to HIGH
        # We need n >= LOW, so n+1 <= HIGH+1
        smooth_nums = generate_smooth(primes, HIGH + 1)
        
        # Filter and sort
        candidates = [x for x in smooth_nums if x >= LOW + 1]
        candidates.sort()
        
        # Check for consecutive pairs
        for i in range(len(candidates) - 1):
            if candidates[i+1] == candidates[i] + 1:
                n = candidates[i]
                if LOW <= n <= HIGH:
                    return n
        return None

    best_n = 10000
    best_score = float('inf') # We want to minimize lpf, so we track the max prime factor
    
    # We don't strictly need to compute lpf for comparison if we iterate prime sets in order
    # The first set that yields a solution has the smallest max prime factor.
    # However, it's possible a solution exists for [2,3,5,7,11,13] but not for [2,3,5,7,11].
    # So iterating in order of max prime is correct.
    
    for primes in prime_sets:
        if time.time() > deadline:
            break
        n = find_best(primes)
        if n is not None:
            # This n has max prime factor = max(primes)
            # Since we iterate from smallest max prime, this is the optimal one found so far.
            # We can return immediately as we want the global minimum.
            # Is it possible that a later set (larger primes) yields a "better" n?
            # No, "better" means smaller largest prime factor.
            # max([2,3,5]) = 5 < max([2,3,5,7]) = 7.
            # So the first success is the global optimum among these sets.
            return n
            
    # If no solution found in the predefined sets, fallback to a heuristic or return 10000
    # Given the density of smooth numbers, a solution for P=13 or P=11 is very likely.
    # But just in case, we can do a quick local search or return 10000.
    # 10000 has lpf 137.
    # A random search might find something better, but smooth number search is exhaustive for small primes.
    
    return 10000
# EVOLVE-BLOCK-END
