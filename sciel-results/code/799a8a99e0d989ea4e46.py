import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Helper to calculate the score for a given n
    def calc_score(n):
        # Generate divisors
        divs = [1]
        temp = n
        p = 2
        # Factorize n to generate divisors efficiently
        # Since n is product of distinct primes (mostly), this is fast
        # But we need to handle exponents if we allow them.
        # For now, assume squarefree or simple exponents.
        # A generic divisor generator:
        factors = []
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
            p += 1 if p == 2 else 2
        if temp > 1:
            factors.append((temp, 1))
            
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Initial candidate: Product of first 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_current = 1
    for p in primes:
        n_current *= p
        
    score_current = calc_score(n_current)
    best_n = n_current
    best_score = score_current
    
    # Local Search: Try swapping the largest prime with a larger one
    # Or try adding a prime and removing a smaller one?
    # Since we are at the limit of 15 primes, we can only swap.
    
    # Generate a list of candidate primes to swap in
    # We need primes > 47
    # Sieve up to some limit
    limit = 1000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    candidate_primes = [i for i in range(48, limit + 1) if is_prime[i]]
    
    # We can also try swapping smaller primes?
    # If we swap 2 with 53, the product increases significantly.
    # n_new = n_current / 2 * 53.
    # If n_current is close to 10^18, we can't increase the product.
    # n_current = 6.14e17.
    # Ratio 53/2 = 26.5. n_new = 1.6e19 > 10^18.
    # So we can only swap if the new prime is smaller than the old prime * (10^18 / n_current).
    # Max ratio allowed = 10^18 / 6.14e17 = 1.62.
    # So we can swap 47 with p such that p < 47 * 1.62 = 76.
    # Primes < 76: 53, 59, 61, 67, 71, 73.
    
    # Let's try all valid swaps for the largest prime first.
    # Then for the second largest, etc.
    
    # To be efficient, we only evaluate if the product is valid.
    
    # We will perform a few iterations of local search.
    # Since calc_score is O(tau(n) log tau(n)) and tau(n) ~ 32768, it's fast.
    # 32768 * log(32768) ~ 32768 * 15 ~ 500k ops.
    # 150s allows for thousands of evaluations.
    
    # We'll try swapping each prime with a larger one, if the product stays <= 10^18.
    # We'll do this for a few rounds.
    
    current_primes = primes[:]
    
    for _ in range(10): # Max 10 rounds of improvement
        improved = False
        # Try swapping each prime p_i with a larger prime q
        for i in range(len(current_primes)):
            p_i = current_primes[i]
            # Find max q such that n * q / p_i <= 10^18
            max_q = int((10**18 * p_i) // n_current)
            if max_q < p_i:
                continue
            
            # Get candidate primes in (p_i, max_q]
            # We have candidate_primes up to 1000.
            # If max_q > 1000, we might miss some, but 1000 is likely enough for small swaps.
            # If we swap a small prime, max_q can be large.
            # e.g. swap 2. max_q = 10^18 * 2 / 6.14e17 = 3.2.
            # So we can swap 2 with 3? No, 3 is already there.
            # Swap 2 with nothing?
            
            # We need to ensure q is not already in current_primes.
            existing_set = set(current_primes)
            
            # Generate candidates from candidate_primes
            # If max_q > 1000, we should extend the sieve or just check primes up to max_q if it's not too big.
            # Given the ratios, max_q is usually small.
            
            for q in candidate_primes:
                if q <= p_i:
                    continue
                if q > max_q:
                    break
                if q in existing_set:
                    continue
                
                # Construct new n
                new_n = n_current // p_i * q
                if new_n > 10**18:
                    continue
                
                # Calculate score
                s = calc_score(new_n)
                
                if s > best_score:
                    best_score = s
                    best_n = new_n
                    current_primes[i] = q
                    n_current = new_n
                    improved = True
                    break # Restart loop with new best? Or continue?
                    # Breaking inner loop, but we should probably restart outer to re-evaluate other swaps
                    # with the new baseline.
            
            if improved:
                break # Restart the round
        
        if not improved:
            break
            
    return best_n
