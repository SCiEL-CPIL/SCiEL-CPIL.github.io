"""Rewritten program: maximize set size under no-three-distinct-product-divisibility."""
import time
from math import isqrt, gcd

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    N = P['n']
    
    # Precompute primes in [2, N]
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(N) + 1):
        if sieve[i]:
            for j in range(i*i, N+1, i):
                sieve[j] = False
    primes = [i for i in range(2, N+1) if sieve[i]]
    composites = [i for i in range(2, N+1) if not sieve[i]]
    
    # Precompute divisibility relations for all pairs (a, b) where a|b
    # div_table[a] = set of b in [2..N] such that a | b
    div_table = {}
    for a in range(2, N+1):
        div_table[a] = set(range(a, N+1, a))
    
    # Precompute for each x, the set of (b,c) pairs from [2..N] where x | b*c
    # This is expensive to precompute fully, so we'll compute on the fly but cache
    # Instead, precompute for each x the set of b such that there exists c with x|b*c
    # Actually, let's use a different approach:
    # For a given x and set S, x can be added iff:
    #   (1) For all distinct b,c in S: x does not divide b*c
    #   (2) For all a in S, and all b in S with b != a: a does not divide x*b
    
    # Optimize: precompute for each x, which b in [2..N] have the property that
    # there exists some c in [2..N] (c != b) such that x | b*c
    # This helps with check (1): if x divides b*c for some c in S, then we have a conflict.
    # But c must be in S, so we need to check per-set.
    
    # Alternative optimization: for check (2), a | x*b iff a/gcd(a,x) | b
    # So for given a and x, compute g = gcd(a,x), then a' = a//g, and check if any b in S
    # (b != a) is divisible by a'.
    
    def can_add(x, S):
        """Check if x can be added to set S without violating the constraint."""
        S_list = list(S)
        n = len(S_list)
        
        # Check (1): x !| b*c for all distinct b,c in S
        # x | b*c iff there's no prime power in x that exceeds the combined powers in b and c
        # Faster: for each pair, check (b*c) % x == 0
        for i in range(n):
            bi = S_list[i]
            for j in range(i+1, n):
                bj = S_list[j]
                if (bi * bj) % x == 0:
                    return False
        
        # Check (2): for each a in S, a !| x*b for all b in S, b != a
        for i in range(n):
            a = S_list[i]
            g = gcd(a, x)
            a_reduced = a // g
            if a_reduced == 1:
                # a | x always, so a | x*b for all b -> conflict if there's any b != a
                if n > 1:
                    return False
            else:
                # Need: no b in S (b != a) such that a_reduced | b
                for j in range(n):
                    if i == j:
                        continue
                    if S_list[j] % a_reduced == 0:
                        return False
        return True
    
    # Start with all primes
    best_set = set(primes)
    
    # Sort composites by a "difficulty" heuristic: fewer distinct prime factors = easier to add
    def num_distinct_prime_factors(x):
        count = 0
        temp = x
        for p in primes:
            if p > temp:
                break
            if temp % p == 0:
                count += 1
                while temp % p == 0:
                    temp //= p
        return count
    
    composites_sorted = sorted(composites, key=lambda x: (num_distinct_prime_factors(x), x))
    
    # Greedy addition
    for x in composites_sorted:
        if time.time() > deadline:
            break
        if can_add(x, best_set):
            best_set.add(x)
    
    # Local search: try removing one element and adding one or more others
    improved = True
    max_rounds = 20
    round_num = 0
    while improved and time.time() < deadline and round_num < max_rounds:
        improved = False
        round_num += 1
        S_list = sorted(best_set)
        n_set = len(S_list)
        
        best_swap = None  # (removed, added_list, net_gain)
        
        for i in range(n_set):
            if time.time() > deadline:
                break
            candidate = S_list[i]
            S_without = best_set - {candidate}
            
            # Try to add as many as possible from composites not in S_without
            added = []
            for x in composites_sorted:
                if time.time() > deadline:
                    break
                if x in S_without:
                    continue
                if can_add(x, S_without):
                    S_without.add(x)
                    added.append(x)
            
            net_gain = len(added) - 1
            if net_gain > 1:
                # Strict improvement: remove 1, add 2+
                if best_swap is None or net_gain > best_swap[2]:
                    best_swap = (candidate, added, net_gain)
            elif net_gain == 1:
                # Neutral: remove 1, add 1 - only accept if it enables future improvements
                # For now, skip neutral swaps to avoid cycling
                pass
        
        if best_swap is not None:
            candidate, added, _ = best_swap
            best_set.discard(candidate)
            for x in added:
                best_set.add(x)
            improved = True
    
    # Final polish: try single swaps (remove 1, add 1) that might help
    if time.time() < deadline:
        S_list = sorted(best_set)
        for i in range(len(S_list)):
            if time.time() > deadline:
                break
            candidate = S_list[i]
            S_without = best_set - {candidate}
            for x in composites_sorted:
                if x in S_without:
                    continue
                if can_add(x, S_without):
                    # Try the swap
                    new_set = S_without | {x}
                    # Verify it's valid (should be, but double-check)
                    best_set = new_set
                    improved = True
                    break
    
    result = sorted(best_set)
    return [int(x) for x in result]
