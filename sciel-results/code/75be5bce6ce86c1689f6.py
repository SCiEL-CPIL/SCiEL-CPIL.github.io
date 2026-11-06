import math
import time
from math import gcd
import random

def _get_primes(limit):
    """Sieve of Eratosthenes."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

# We need primes up to a reasonably large number to allow swaps.
# Max prime in product < 10^18. If we have 14 primes, the 15th can be up to 10^18 / P_14 ~ 10^18 / 10^16 = 100.
# If we have 10 primes, the 11th can be up to 10^18 / P_10 ~ 10^18 / 2000 = 5*10^14.
# Let's get primes up to 10^6 to be safe for swaps.
_PRIMES = _get_primes(1000000)

def _tau_perp_fast(n, factorization):
    """Compute tau_perp using factorization for speed.
    factorization: list of (prime, exponent)
    """
    # Generate all divisors from factorization
    divs = [1]
    for p, e in factorization:
        new_divs = []
        for d in divs:
            val = d
            for exp in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _search():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    LIMIT = 10**18
    
    # Initial candidate: Primorial of first 15 primes
    # P_15 = 614889782588491410
    initial_primes = _PRIMES[:15]
    best_n = 1
    best_factorization = []
    for p in initial_primes:
        best_n *= p
        best_factorization.append((p, 1))
    
    best_score = _tau_perp_fast(best_n, best_factorization)
    
    # Local Search: Try swapping primes
    # We maintain a list of (prime, exponent). Initially all exponents 1.
    # We try to replace p_i with a different prime q.
    # To keep the number of divisors high, we prefer square-free.
    
    # Helper to check if a list of (p, e) is valid (product <= LIMIT)
    def is_valid(factorization):
        prod = 1
        for p, e in factorization:
            prod *= p**e
            if prod > LIMIT:
                return False
        return True

    def get_score(factorization):
        if not is_valid(factorization):
            return -1
        n = 1
        for p, e in factorization:
            n *= p**e
        return _tau_perp_fast(n, factorization)

    # 1. Try swapping each prime with a larger prime
    # We only consider primes larger than the current one to maintain "sorted" logic implicitly?
    # Actually, order doesn't matter for the number, but for iteration we need to be careful.
    
    # Pre-compute prime indices for fast lookup?
    # We will just iterate through _PRIMES.
    
    # Optimization: Only try swaps that keep the product under LIMIT.
    # If we replace p_i with q, new_prod = old_prod * q / p_i.
    # So q <= LIMIT * p_i / old_prod.
    
    current_factorization = best_factorization[:]
    current_n = best_n
    current_score = best_score
    
    # Iterate a few times to allow multiple swaps?
    # Or just one pass of hill climbing.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(len(current_factorization)):
            p_i, e_i = current_factorization[i]
            # Try replacing p_i with a larger prime q
            # Max q allowed:
            max_q = LIMIT * p_i // current_n
            
            # Find primes in _PRIMES that are > p_i and <= max_q
            # We can scan _PRIMES
            # To avoid O(N^2) complexity, we can just check a few random ones or a range.
            # Since _PRIMES is sorted, we can binary search start index.
            
            # For simplicity and speed, let's just try the next few primes or random primes in range.
            # But we must ensure we don't pick a prime already in the factorization.
            used_primes = {p for p, _ in current_factorization}
            
            # Generate candidate primes
            # If max_q is large, we can pick random primes.
            # If max_q is close to p_i, we pick the next few.
            
            # Let's try the next 10 primes after p_i, and 10 random primes in (p_i, max_q]
            start_idx = _PRIMES.index(p_i) + 1
            
            candidates = []
            # Next 10 primes
            for j in range(start_idx, min(start_idx + 10, len(_PRIMES))):
                if _PRIMES[j] <= max_q:
                    candidates.append(_PRIMES[j])
                else:
                    break
            
            # Random primes
            if max_q > _PRIMES[-1]:
                # All primes are valid
                pass
            else:
                end_idx = len(_PRIMES)
                for j in range(len(_PRIMES)-1, -1, -1):
                    if _PRIMES[j] <= max_q:
                        end_idx = j + 1
                        break
                
                # Pick 10 random indices between start_idx and end_idx
                if end_idx > start_idx:
                    for _ in range(10):
                        idx = random.randint(start_idx, end_idx - 1)
                        candidates.append(_PRIMES[idx])
            
            # Evaluate candidates
            for q in candidates:
                if q in used_primes:
                    continue
                if q <= p_i: # We want to explore larger space, but smaller might be valid if we removed a larger one? 
                             # But we are replacing p_i with q. If q < p_i, product decreases.
                             # This allows fitting MORE primes later?
                             # For now, let's stick to swapping with larger to see if structure improves.
                             # Actually, swapping with smaller is also valid.
                             # Let's allow any q != p_i.
                             pass
                
                new_factorization = current_factorization[:]
                new_factorization[i] = (q, e_i)
                
                # Check validity
                new_n = current_n // p_i * q
                if new_n > LIMIT:
                    continue
                
                score = _tau_perp_fast(new_n, new_factorization)
                if score > current_score:
                    current_score = score
                    current_n = new_n
                    current_factorization = new_factorization
                    best_score = score
                    best_n = new_n
                    best_factorization = new_factorization
                    improved = True
                    # Break to restart the outer loop with the new base?
                    # Or continue?
                    break # Break inner candidate loop
            
            if time.time() > deadline - 10:
                break

    # 2. Try adding a prime if space allows
    # This is hard to do efficiently in the same loop.
    # But if we have < 15 primes, we can try adding the smallest available prime.
    
    # 3. Try increasing exponents
    # If we have space, try p_i^2.
    for i in range(len(current_factorization)):
        p_i, e_i = current_factorization[i]
        if e_i == 1:
            new_n = current_n * p_i
            if new_n <= LIMIT:
                new_factorization = current_factorization[:]
                new_factorization[i] = (p_i, 2)
                score = _tau_perp_fast(new_n, new_factorization)
                if score > best_score:
                    best_score = score
                    best_n = new_n
                    best_factorization = new_factorization
                    current_n = new_n
                    current_factorization = new_factorization
                    current_score = score

    return best_n

def construct():
    return _search()
