import time
import math

def construct():
    deadline = time.time() + 80.0
    TARGET = 123457

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # We need primes up to a point where smooth numbers cover the target
    # 123457 is not too large. Primes up to 10000 are more than enough.
    primes_all = sieve(10000)
    
    # Fallback if no pair is found with small primes (should not happen)
    best_result = [1, TARGET - 1]
    
    def generate_smooth_numbers(primes, limit):
        """
        Generate all numbers <= limit whose prime factors are all in primes.
        Returns a set of such numbers.
        """
        if not primes:
            return {1}
        
        smooth = {1}
        for p in primes:
            new_smooth = set(smooth)
            # Multiply existing numbers by powers of p
            # We iterate over a snapshot of smooth to avoid using newly added numbers in the same prime step
            for x in list(smooth):
                val = x
                while val * p <= limit:
                    val *= p
                    new_smooth.add(val)
            smooth = new_smooth
        return smooth

    # Iterate over increasing sets of primes
    for k in range(1, len(primes_all) + 1):
        if time.time() > deadline - 5:
            break

        primes_subset = primes_all[:k]
        
        # Generate all primes-smooth numbers up to TARGET
        smooth_set = generate_smooth_numbers(primes_subset, TARGET)
        
        # Check for complementary pair
        found_pair = None
        for a in smooth_set:
            b = TARGET - a
            if b < 1:
                continue
            if b in smooth_set:
                found_pair = (a, b)
                break
        
        if found_pair:
            # The minimal max prime factor is primes_subset[-1]
            # We return the pair
            return list(found_pair)

    return best_result
