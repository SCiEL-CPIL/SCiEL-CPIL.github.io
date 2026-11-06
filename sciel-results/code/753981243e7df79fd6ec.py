"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START

def _prime_factors(n):
    """Return set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s kill
    
    N = 512
    
    # Precompute prime factors for all numbers 1..511
    pf = [set() for _ in range(N)]
    for a in range(1, N):
        pf[a] = _prime_factors(a)
    
    # Precompute contribution values as Fractions for exact comparison
    contrib = {a: Fraction(1, N - a) for a in range(1, N)}
    
    def conflicts_with_set(a, chosen_set):
        """Check if a conflicts (shares a prime factor) with any element in chosen_set."""
        if a == 1:
            return False
        for b in chosen_set:
            if b == 1:
                continue
            # Check if any prime factor is shared
            for p in pf[a]:
                if p in pf[b]:
                    return True
        return False
    
    def score_of(chosen):
        """Compute exact score as Fraction."""
        total = Fraction(0, 1)
        for a in chosen:
            total += contrib[a]
        return total
    
    # Greedy initialization: pick numbers in order of highest contribution (largest a)
    # that don't conflict with already chosen numbers.
    chosen = set()
    for a in range(N - 1, 0, -1):
        if not conflicts_with_set(a, chosen):
            chosen.add(a)
    
    best_score = score_of(chosen)
    best_chosen = set(chosen)
    unchosen = set(range(1, N)) - chosen
    
    # Local search with 1-for-1 swaps
    while time.time() < DEADLINE:
        improved = False
        
        # Try 1-for-1 swaps: remove one chosen, add one unchosen
        chosen_list = list(chosen)
        for a in chosen_list:
            if time.time() > DEADLINE:
                break
            
            val_a = contrib[a]
            chosen.remove(a)
            unchosen.add(a)
            
            # Try adding any unchosen that fits
            found_swap = False
            for b in sorted(unchosen, key=lambda x: -contrib[x]):
                if b == a:
                    continue
                if not conflicts_with_set(b, chosen):
                    new_score = best_score - val_a + contrib[b]
                    if new_score > best_score:
                        chosen.add(b)
                        unchosen.remove(b)
                        best_score = new_score
                        best_chosen = set(chosen)
                        improved = True
                        found_swap = True
                        break
            
            # Restore a if we didn't swap it out
            if not found_swap:
                chosen.add(a)
                unchosen.remove(a)
        
        # Try adding any unchosen number that doesn't conflict (if possible)
        for b in sorted(unchosen, key=lambda x: -contrib[x]):
            if time.time() > DEADLINE:
                break
            if not conflicts_with_set(b, chosen):
                new_score = best_score + contrib[b]
                if new_score > best_score:
                    chosen.add(b)
                    unchosen.remove(b)
                    best_score = new_score
                    best_chosen = set(chosen)
                    improved = True
        
        if not improved:
            break
    
    # Convert to list for return
    return list(best_chosen)

# EVOLVE-BLOCK-END
