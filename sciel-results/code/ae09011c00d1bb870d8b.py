import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin

    # 1. Generate a list of small primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # We need primes up to a reasonably large number to allow swaps.
    # The product of the first 15 primes is ~6e17.
    # We can swap larger primes with smaller ones to fit more primes or rearrange.
    # Let's generate primes up to 200,000 to have plenty of candidates for the largest slot.
    # 10^18 / (product of first 14 primes) ~ 10^18 / 1.3e15 ~ 760.
    # Actually, product of first 14 is 2*3*...*43 = 1311359619.
    # 10^18 / 1311359619 ~ 762,559,740.
    # So we might need primes up to ~760 million? No, that's too many for a simple sieve in Python within time limits if we just want to list them.
    # However, we only need to *swap* one prime. 
    # If we replace the largest prime $p_{15}$, we can go up to $10^{18} / \text{prod}(p_1 \dots p_{14}) \approx 7.6 \times 10^8$.
    # Generating primes up to 10^9 is too slow for a simple sieve in Python (memory and time).
    # But we don't need *all* primes. We just need *some* good candidates.
    # The score depends on the structure. Smaller primes are generally better.
    # Let's generate primes up to 1,000,000. This is fast.
    # If the optimal replacement is a prime > 10^6, we might miss it, but it's unlikely to be significantly better than a prime < 10^6 
    # because smaller primes create "denser" divisor lattices which tend to have more coprime adjacencies.
    
    primes = sieve(1000000)
    prime_set = set(primes)
    
    # 2. Define tau_perp calculation optimized for speed
    def tau_perp(prime_factors):
        # prime_factors is the list of distinct primes dividing n
        # Generate divisors
        divs = [1]
        for p in prime_factors:
            # Optimization: extend in place or use list comprehension
            # List comprehension is usually faster in Python
            divs = divs + [d * p for d in divs]
            
        divs.sort()
        
        # Count adjacent coprime pairs
        score = 0
        # Use local variables for speed
        gcd = math.gcd
        for i in range(len(divs) - 1):
            if gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # 3. Initialize with the "greedy" baseline: product of the smallest primes
    # Max k is 15.
    max_k = 15
    initial_primes = primes[:max_k]
    
    current_primes = list(initial_primes)
    current_n = 1
    for p in current_primes:
        current_n *= p
        
    best_n = current_n
    best_score = tau_perp(current_primes)
    current_score = best_score
    
    # 4. Local Search
    # We maintain the set of primes.
    # We try to swap each prime p_i with a prime q not in the set.
    # Constraint: (current_n // p_i) * q <= 10**18
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        current_prime_set = set(current_primes)
        
        # Shuffle indices to randomize search order
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5.0:
                return best_n
                
            p_old = current_primes[idx]
            
            # Max value for new prime
            # new_n = (current_n // p_old) * p_new <= 10**18
            # p_new <= 10**18 * p_old // current_n
            max_p_new = (10**18 * p_old) // current_n
            
            # Find best candidate for this slot
            best_candidate = None
            best_candidate_score = current_score - 1 # Must be strictly better to swap
            
            # We scan through primes. 
            # Optimization: We can skip primes already in the set.
            # We can also limit the range if we want, but 10^6 is fine.
            for p_new in primes:
                if p_new in current_prime_set:
                    continue
                if p_new > max_p_new:
                    break
                
                new_primes = current_primes.copy()
                new_primes[idx] = p_new
                
                s_new = tau_perp(new_primes)
                
                if s_new > best_candidate_score:
                    best_candidate_score = s_new
                    best_candidate = p_new
            
            if best_candidate is not None:
                # Apply the best swap
                current_primes[idx] = best_candidate
                current_n = (current_n // p_old) * best_candidate
                current_score = best_candidate_score
                best_score = current_score
                best_n = current_n
                improved = True
                # Break to restart with new state? 
                # Yes, because the set changed.
                break

    return best_n
