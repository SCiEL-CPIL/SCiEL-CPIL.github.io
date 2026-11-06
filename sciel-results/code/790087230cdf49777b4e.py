import math
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin before 120s limit
    
    LIMIT = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    candidates = []

    def backtrack(idx, current_n, current_tau, max_exp):
        if time.time() > DEADLINE:
            return
        
        # If we have a valid number (n >= 2), add it
        if current_n >= 2:
            candidates.append((current_tau, current_n))
        
        if idx >= len(primes):
            return
            
        p = primes[idx]
        val = 1
        # Try exponents from max_exp down to 0
        for e in range(max_exp, -1, -1):
            if current_n * val > LIMIT:
                break
            
            new_tau = current_tau * (e + 1)
            new_n = current_n * val
            
            # Recurse to next prime
            # The next prime's exponent cannot exceed current exponent e
            backtrack(idx + 1, new_n, new_tau, e)
            
            val *= p
            if val > LIMIT // current_n:
                break

    # Start search
    # Max exponent for 2 is such that 2^e <= 10^18 => e <= 59
    backtrack(0, 1, 1, 60)
    
    if not candidates:
        # Fallback to seed if search failed or timed out immediately
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19

    # Sort by tau descending
    candidates.sort(key=lambda x: -x[0])
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for tau, n in candidates:
        if n not in seen:
            seen.add(n)
            unique_candidates.append((tau, n))
            if len(unique_candidates) >= 100:
                break

    def get_divisors_from_factors(factors):
        divisors = [1]
        for p, e in factors:
            new_divs = []
            for d in divisors:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        divisors.sort()
        return divisors

    def factorize_small_primes(n, primes):
        factors = []
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                exp = 0
                while n % p == 0:
                    n //= p
                    exp += 1
                factors.append((p, exp))
        if n > 1:
            factors.append((n, 1))
        return factors

    def compute_score(n, primes):
        factors = factorize_small_primes(n, primes)
        divs = get_divisors_from_factors(factors)
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    best_n = unique_candidates[0][1]
    best_score = -1
    
    # Evaluate top candidates
    for tau, n in unique_candidates:
        if time.time() > DEADLINE:
            break
        
        try:
            score = compute_score(n, primes)
            if score > best_score:
                best_score = score
                best_n = n
        except Exception:
            continue
            
    return best_n

# EVOLVE-BLOCK-END
