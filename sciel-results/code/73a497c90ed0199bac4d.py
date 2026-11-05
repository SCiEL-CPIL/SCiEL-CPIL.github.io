import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n, with P(1)=1.
    
    Strategy:
    Instead of sieving all numbers up to N and scanning, we generate K-smooth numbers 
    for increasing K (smallest prime factor limit).
    We start with K=2, then K=3, etc.
    For a given K, we generate all numbers <= N that are K-smooth.
    We check if any a in this set has N-a also in the set.
    The first K for which such a pair exists gives the minimal max(P(a), P(b)).
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # seconds

    # First few primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We will iterate through the number of primes to include
    # k=1 means only prime 2
    # k=2 means primes 2, 3
    # etc.
    
    # Helper to generate K-smooth numbers up to limit
    def generate_smooth(primes_subset, limit):
        smooths = {1}
        # Iterative generation
        # Start with 1, multiply by primes
        # This simple DFS/BFS might be slow if not careful, but for small K and N=1e5 it's fine.
        # Better: dynamic programming or recursive generation.
        
        # Recursive generation
        def gen(idx, current_val):
            if current_val > limit:
                return
            if idx == len(primes_subset):
                return
            p = primes_subset[idx]
            # Multiply by p^0, p^1, p^2...
            val = current_val
            while val <= limit:
                smooths.add(val)
                # Recurse to next prime
                gen(idx + 1, val)
                val *= p
                if val > limit:
                    break
            # Note: The loop above adds val *before* multiplying for next power.
            # Wait, the logic above is slightly mixed.
            # Let's do it cleanly.
            
        # Let's use a cleaner iterative approach for small N.
        # Initialize set with 1
        # For each prime p in subset:
        #   new_smooths = set()
        #   for s in current_smooths:
        #       v = s
        #       while v * p <= limit:
        #           v *= p
        #           new_smooths.add(v)
        #   current_smooths.update(new_smooths)
        
        current_smooths = {1}
        for p in primes_subset:
            new_smooths = set()
            for s in current_smooths:
                v = s
                while v * p <= limit:
                    v *= p
                    new_smooths.add(v)
            current_smooths.update(new_smooths)
            
        return current_smooths

    # Iterate K from 1 to len(primes)
    for k in range(1, len(primes) + 1):
        if time.time() - start_time > DEADLINE:
            # Fallback to a known good pair or simple heuristic if timeout
            # But this should finish in ms.
            # Return a pair with small primes, e.g., 1 and TARGET-1
            return [1, TARGET - 1]
            
        current_primes = primes[:k]
        smooth_set = generate_smooth(current_primes, TARGET)
        
        # Check for pair
        for a in smooth_set:
            b = TARGET - a
            if b > 0 and b in smooth_set:
                return [a, b]
                
    # If no pair found with these primes (unlikely), fallback
    # Find a pair using a simple sieve for the rest
    # But for N=123457, the optimal K is small (likely < 15).
    # So this loop will terminate early.
    
    # Fallback: Just return 1, TARGET-1
    return [1, TARGET - 1]
