"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import isqrt

def _sieve_primes(n):
    """Return list of primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _factorize(n, primes):
    """Return dict of prime->exponent for n."""
    factors = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            factors[p] = exp
    if n > 1:
        factors[n] = 1
    return factors

def construct():
    n = 256
    primes = _sieve_primes(n)
    
    # Compute prime factor dicts for each number in 2..n
    prime_factors = {}
    for i in range(2, n + 1):
        prime_factors[i] = _factorize(i, primes)
    
    # Precompute divisibility: divs[x][y] is True if x divides y
    # We only need this for x, y in 2..n
    # But checking a|bc is not just divs.
    # We will stick to the modular check but optimize the loop.
    
    def can_add(a, S_list):
        """Check if a can be added to list S_list without violating the condition."""
        m = len(S_list)
        # Check 1: a does not divide b*c for any distinct b,c in S
        # Check 2: b does not divide a*c for any distinct b,c in S
        # Check 3: c does not divide a*b for any distinct b,c in S
        # Note: Check 2 and 3 are symmetric in b,c. 
        # If we iterate all pairs (b,c), we can check:
        # 1. a % (b*c) == 0 ? No, (b*c) % a == 0
        # 2. (a*c) % b == 0
        # 3. (a*b) % c == 0
        
        # Optimization: 
        # If m is small, just do the loops.
        # We can break early.
        
        for i in range(m):
            b = S_list[i]
            for j in range(i + 1, m):
                c = S_list[j]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
        return True

    candidates = list(range(2, n + 1))
    
    def run_greedy(order):
        S = []
        for a in order:
            if can_add(a, S):
                S.append(a)
        return S

    # Strategy 1: Deterministic Orderings
    def sort_key_lpf(x):
        pf = prime_factors[x]
        largest_pf = max(pf.keys()) if pf else 0
        num_pf = len(pf)
        return (-largest_pf, num_pf, -x)
        
    def sort_key_npf(x):
        pf = prime_factors[x]
        largest_pf = max(pf.keys()) if pf else 0
        num_pf = len(pf)
        return (-num_pf, -largest_pf, -x)

    def sort_key_prime_first(x):
        pf = prime_factors[x]
        is_prime = len(pf) == 1
        largest_pf = max(pf.keys()) if pf else 0
        return (0 if is_prime else 1, -largest_pf, -x)
    
    orderings = [
        sorted(candidates, key=sort_key_lpf),
        sorted(candidates, key=sort_key_npf),
        sorted(candidates, key=lambda x: -x),
        sorted(candidates, key=sort_key_prime_first),
        sorted(candidates, key=lambda x: x),
        sorted(candidates, key=lambda x: -len(prime_factors[x])),
    ]
    
    best_set = []
    for o in orderings:
        S = run_greedy(o)
        if len(S) > len(best_set):
            best_set = S
    
    # Strategy 2: Randomized Greedy with Restarts
    # Use a fixed seed for reproducibility if needed, but random is fine.
    # Limit number of restarts to stay within time limit (60s).
    # 256 elements, can_add is O(k^2). k~20. 400 ops per add. 256 adds. ~100k ops per run.
    # 100 runs is ~10M ops. Python might take ~1-2s. Safe.
    
    random.seed(42)
    for _ in range(50):
        random.shuffle(candidates)
        S = run_greedy(candidates)
        if len(S) > len(best_set):
            best_set = S
            
    # Strategy 3: Local Search on best_set
    # Try to swap elements to allow additions.
    # This is heuristic.
    # We'll do a simple hill climbing: 
    # 1. Try to add any x not in best_set.
    # 2. If fail, find a minimal blocking set? Too complex.
    # 3. Try swapping u in best_set with v not in best_set if it doesn't reduce size (it doesn't) 
    #    but we need to check if the new set allows adding a third element w.
    
    # Simpler Local Search:
    # Iterate through all pairs (u in S, v not in S).
    # If S' = (S \ {u}) U {v} is valid (it is, since we just swapped, but we must check validity of the new set? 
    # Actually, if S is valid, S' might not be valid. We must check can_add(v, S\{u}) is True.
    # If S' is valid, we keep it temporarily? No, we want to increase size.
    # So:
    # For each u in S:
    #   For each v not in S:
    #     If can_add(v, S \ {u}):
    #        S_temp = (S \ {u}) U {v}
    #        # Now try to add any w not in S_temp
    #        for w in candidates:
    #           if w not in S_temp and can_add(w, S_temp):
    #               S_new = S_temp U {w}
    #               if len(S_new) > len(best_set):
    #                   best_set = S_new
    #               break # Found an improvement, restart search?
    
    # This is O(N^3 * Cost(can_add)). N=256. N^3 is 16M. Too slow for Python if Cost is high.
    # We can limit this.
    
    # Let's just do a few random restarts and pick the best.
    # The current score is 1.05. 
    # Let's try to improve the ordering keys.
    
    # A good heuristic for "no two other product divides" is to pick numbers with large prime factors 
    # because they are less likely to divide products of smaller numbers?
    # Actually, if a has a large prime factor p, then for a|bc, p must divide b or c.
    # So if we pick numbers that are "coprime" or have distinct large primes, we reduce conflicts.
    
    # Let's add an ordering based on the product of prime factors (the number itself) 
    # but prioritizing those with few divisors in the set?
    
    # We will return the best found so far.
    return list(best_set)
# EVOLVE-BLOCK-END
