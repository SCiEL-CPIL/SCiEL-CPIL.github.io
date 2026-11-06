"""Combined program: Maximize coprime adjacent divisor pairs using optimized local search."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _primes_up_to(limit):
    """Generate primes up to limit using sieve of Eratosthenes."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

# Prime pool for search. 
# We need primes up to at least 1000 to allow for larger replacements.
_PRIMES = _primes_up_to(1000)
# Create a set for fast lookup
_PRIME_SET = set(_PRIMES)

def _tau_perp_bitmask(primes, exponents):
    """
    Compute tau_perp score using bitmask representation for speed.
    """
    if not primes:
        return 0
        
    divisors = [(1, 0)]
    
    # Pre-calculate powers to avoid repeated multiplication in inner loop?
    # Actually, the current method is fine.
    
    for idx, (p, e) in enumerate(zip(primes, exponents)):
        if e == 0:
            continue
        bit = 1 << idx
        current = list(divisors)
        pk = 1
        for k in range(1, e + 1):
            pk *= p
            for d_val, d_mask in current:
                divisors.append((d_val * pk, d_mask | bit))
    
    # Sort by value
    divisors.sort(key=lambda x: x[0])
    
    if len(divisors) < 2:
        return 0
    
    count = 0
    for i in range(len(divisors) - 1):
        if (divisors[i][1] & divisors[i+1][1]) == 0:
            count += 1
    return count

def _compute_n(primes, exponents):
    """Compute n from prime factorization."""
    n = 1
    for p, e in zip(primes, exponents):
        if e > 0:
            n *= p ** e
    return n

def construct():
    """
    Returns an integer n <= 10^18 that maximizes the number of adjacent 
    coprime pairs in its sorted list of divisors.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    max_n = 10**18
    
    best_score = 0
    best_n = 2
    
    # Generate candidate base states: primorials of first k primes
    bases = []
    prod = 1
    for i, p in enumerate(_PRIMES):
        if prod * p > max_n:
            break
        prod *= p
        bases.append((i + 1, prod))
    
    # Search from the top few bases (largest primorials first)
    # Also include a few smaller ones in case removing a prime allows better structure
    search_bases = bases[-5:]  # Last 5 in the list
    
    # Also add a random start point? No, stick to structured starts first.
    # Maybe add a start point with fewer primes but larger values?
    # For now, rely on local search to move primes.
    
    for k, base_prod in search_bases:
        if time.time() > deadline:
            break
            
        # Initial state: first k primes, exponent 1 each
        current_primes = _PRIMES[:k]
        current_exps = [1] * k
        current_n = base_prod
        current_score = _tau_perp_bitmask(current_primes, current_exps)
        
        if current_score > best_score:
            best_score = current_score
            best_n = current_n
        
        # Local Search
        improved = True
        iterations = 0
        max_iterations = 100  # Reduced iterations to allow more bases if needed, or just faster convergence
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Move 1: Increase exponent of prime i
            for i in range(k):
                if time.time() > deadline:
                    break
                new_exps = current_exps[:]
                new_exps[i] += 1
                new_n = current_n * current_primes[i]
                if new_n <= max_n:
                    score = _tau_perp_bitmask(current_primes, new_exps)
                    if score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
            
            # Move 2: Decrease exponent of prime i (if > 1)
            for i in range(k):
                if time.time() > deadline:
                    break
                if current_exps[i] <= 1:
                    continue
                new_exps = current_exps[:]
                new_exps[i] -= 1
                new_n = current_n // current_primes[i]
                if new_n >= 2:
                    score = _tau_perp_bitmask(current_primes, new_exps)
                    if score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
            
            # Move 3: Swap exponents between primes i and j
            # This is less useful if exponents are all 1, but good if mixed
            for i in range(k):
                if time.time() > deadline:
                    break
                if current_exps[i] <= 0:
                    continue
                for j in range(k):
                    if i == j:
                        continue
                    new_exps = current_exps[:]
                    new_exps[i] -= 1
                    new_exps[j] += 1
                    new_n = (current_n // current_primes[i]) * current_primes[j]
                    if new_n <= max_n:
                        score = _tau_perp_bitmask(current_primes, new_exps)
                        if score > current_score:
                            current_exps = new_exps
                            current_n = new_n
                            current_score = score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n
            
            # Move 4: Replace prime at position i with a different prime from the pool
            # Optimization: Only consider primes that keep n <= max_n
            # and are not already in the set
            for i in range(k):
                if time.time() > deadline:
                    break
                if current_exps[i] <= 0:
                    continue
                old_p = current_primes[i]
                exp_i = current_exps[i]
                # We need new_p such that n * (new_p / old_p) <= max_n
                # n * new_p <= max_n * old_p
                # new_p <= (max_n * old_p) / n
                limit_p = (max_n * old_p) // current_n
                
                # Find candidates in _PRIMES
                # To speed up, we can binary search or just iterate and break
                # Since _PRIMES is sorted, we can find the index
                # But iterating is fine if we break early
                for new_p in _PRIMES:
                    if new_p > limit_p:
                        break
                    if new_p == old_p:
                        continue
                    if new_p in current_primes:
                        continue
                    
                    new_primes = current_primes[:]
                    new_primes[i] = new_p
                    new_n = (current_n // old_p) * new_p
                    if new_n <= max_n:
                        score = _tau_perp_bitmask(new_primes, current_exps)
                        if score > current_score:
                            current_primes = new_primes
                            current_n = new_n
                            current_score = score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n
                            # Don't break immediately? 
                            # Taking the first improvement is standard hill climbing.
                            # Breaking allows exploring other positions in the same iteration.
                            # But if we don't break, we might evaluate many bad moves.
                            # Let's break to save time.
                            break
            
            # Move 5: Add a new prime
            # Try adding a prime with exponent 1
            # Find smallest prime not in current_primes such that n * p <= max_n
            # Actually, any prime. Let's try a few.
            # This increases k by 1.
            if k < 20: # Limit number of primes to keep divisor count manageable
                # Try adding the smallest available prime
                # Or a prime that fits
                for new_p in _PRIMES:
                    if new_p in current_primes:
                        continue
                    if current_n * new_p > max_n:
                        break
                    new_primes = current_primes + [new_p]
                    new_exps = current_exps + [1]
                    new_n = current_n * new_p
                    score = _tau_perp_bitmask(new_primes, new_exps)
                    if score > current_score:
                        current_primes = new_primes
                        current_exps = new_exps
                        current_n = new_n
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                        break # Add one at a time
            
            # Move 6: Remove a prime
            # This decreases k by 1.
            # Useful if we have too many primes and want to increase exponents of others?
            # Or if removing a "bad" prime improves the structure.
            for i in range(k):
                if time.time() > deadline:
                    break
                if current_exps[i] <= 0:
                    continue
                # Only consider removing if exp is 1? 
                # If exp > 1, reducing exp is handled by Move 2.
                # Removing a prime with exp 1 is distinct.
                if current_exps[i] > 1:
                    continue
                    
                new_primes = current_primes[:i] + current_primes[i+1:]
                new_exps = current_exps[:i] + current_exps[i+1:]
                new_n = current_n // current_primes[i]
                if new_n >= 2:
                    score = _tau_perp_bitmask(new_primes, new_exps)
                    if score > current_score:
                        current_primes = new_primes
                        current_exps = new_exps
                        current_n = new_n
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                        break # Remove one at a time
        
        # After local search from this base, check time
        if time.time() > deadline:
            break
    
    # Final verification: ensure best_n is valid
    if best_n < 2:
        best_n = 2
    
    return best_n
# EVOLVE-BLOCK-END
