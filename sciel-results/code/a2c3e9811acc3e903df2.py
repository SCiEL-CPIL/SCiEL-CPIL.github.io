import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    moduli = list(range(2, 129))
    n_moduli = len(moduli)
    
    # Precompute prime factors for each modulus
    # and smallest prime factor
    prime_factors = {}
    smallest_prime_factor = {}
    
    def get_prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                if d not in factors:
                    factors.append(d)
                n //= d
            d += 1
        if n > 1:
            if n not in factors:
                factors.append(n)
        return factors

    for m in moduli:
        pf = get_prime_factors(m)
        prime_factors[m] = pf
        smallest_prime_factor[m] = min(pf)
        
    # Sort moduli by smallest prime factor descending
    # This prioritizes numbers with large prime factors (easiest to place)
    sorted_moduli = sorted(moduli, key=lambda x: smallest_prime_factor[x], reverse=True)
    
    # Greedy Selection
    used_slots = {} # prime -> count used
    selected_moduli = []
    
    def try_greedy(order):
        used = {}
        sel = []
        for m in order:
            pf = prime_factors[m]
            can_add = True
            for p in pf:
                if used.get(p, 0) >= p:
                    can_add = False
                    break
            if can_add:
                sel.append(m)
                for p in pf:
                    used[p] = used.get(p, 0) + 1
        return sel

    best_set = try_greedy(sorted_moduli)
    best_size = len(best_set)
    
    # Try other orderings
    # Sort by number of prime factors ascending (fewer factors = less constraint)
    order_by_num_pf = sorted(moduli, key=lambda x: len(prime_factors[x]))
    s2 = try_greedy(order_by_num_pf)
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2
        
    # Sort by value descending
    order_desc = sorted(moduli, reverse=True)
    s3 = try_greedy(order_desc)
    if len(s3) > best_size:
        best_size = len(s3)
        best_set = s3
        
    # Random restarts
    for _ in range(500):
        if time.time() > deadline:
            break
        random.shuffle(sorted_moduli)
        s = try_greedy(sorted_moduli)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Local Search: Try to swap moduli to increase count
    # This is complex, so we stick with strong greedy.
    # However, we can try to add any remaining moduli if slots open up?
    # No, slots only fill up.
    
    # Compute Residues
    # For each selected modulus m, we need to assign residues v_{m,p} for each p|m
    # such that for each p, v_{m,p} are distinct for all m in S_p.
    
    # Group selected moduli by prime
    moduli_by_prime = {p: [] for p in prime_factors[best_set[0]]} # Init dict
    # Better: collect all primes involved
    all_primes = set()
    for m in best_set:
        for p in prime_factors[m]:
            all_primes.add(p)
            
    moduli_by_prime = {p: [] for p in all_primes}
    for m in best_set:
        for p in prime_factors[m]:
            moduli_by_prime[p].append(m)
            
    # Assign values
    # v[m][p] = value
    v = {}
    for m in best_set:
        v[m] = {}
        
    for p in all_primes:
        ms = moduli_by_prime[p]
        # Assign distinct values 0..len(ms)-1
        # Shuffle to diversify? No, deterministic is fine for now.
        # To ensure CRT works, any distinct values work.
        for i, m in enumerate(ms):
            v[m][p] = i
            
    # Compute final residues r_m using CRT
    result = []
    for m in best_set:
        pf = prime_factors[m]
        if not pf:
            r = 0
        else:
            # Solve r ≡ v[m][p] (mod p) for all p in pf
            # Use CRT
            # Since p are distinct primes, we can use iterative CRT
            r = 0
            M = 1
            for p in pf:
                a = v[m][p]
                # Solve x ≡ r (mod M) and x ≡ a (mod p)
                # x = r + k*M
                # r + k*M ≡ a (mod p)
                # k*M ≡ a - r (mod p)
                # k ≡ (a - r) * M^{-1} (mod p)
                # Find M^{-1} mod p
                # M and p are coprime
                inv_M = pow(M, -1, p) # Python 3.8+
                k = ((a - r) * inv_M) % p
                r = r + k * M
                M *= p
                r %= M
            r = r % m
        result.append([r, m])
        
    if not result:
        result = [[1, 2]]
        
    return result
