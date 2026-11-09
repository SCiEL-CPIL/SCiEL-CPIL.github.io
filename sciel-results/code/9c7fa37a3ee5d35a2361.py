"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import lru_cache

def _check_pair(r1, m1, r2, m2):
    """Return True if classes are disjoint (gcd(m1,m2) does not divide r1-r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid_set(classes):
    """Check if all pairs are disjoint."""
    n = len(classes)
    for i in range(n):
        for j in range(i+1, n):
            r1, m1 = classes[i]
            r2, m2 = classes[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _find_max_classes(max_mod=128, deadline=None):
    """
    Multi-start randomized greedy search with local improvement for maximum pairwise-disjoint residue classes.
    """
    if deadline is None:
        deadline = time.time() + 85  # safety margin before 90s kill
    
    best_set = []
    best_size = 0
    
    # Pre-compute gcds for all pairs of moduli up to max_mod
    gcd_table = [[0]*(max_mod+1) for _ in range(max_mod+1)]
    for m1 in range(2, max_mod+1):
        for m2 in range(2, max_mod+1):
            gcd_table[m1][m2] = gcd(m1, m2)
            
    all_moduli = list(range(2, max_mod+1))
    
    def _try_add(modulus, current_set, gcd_table):
        """Try to add modulus to current_set, returning new set or None."""
        # Check coprime constraint first
        for r_old, m_old in current_set:
            if gcd_table[modulus][m_old] == 1:
                return None
        
        # Find a residue r for modulus that works with all current_set
        for r in range(modulus):
            ok = True
            for r_old, m_old in current_set:
                g = gcd_table[modulus][m_old]
                if (r - r_old) % g == 0:
                    ok = False
                    break
            if ok:
                return current_set + [[r, modulus]]
        return None
    
    def _local_search(current_set, gcd_table, deadline):
        """Try to improve current_set by adding more moduli."""
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            # Try adding each unused modulus
            used_mods = {m for _, m in current_set}
            for m in all_moduli:
                if m in used_mods:
                    continue
                new_set = _try_add(m, current_set, gcd_table)
                if new_set is not None:
                    current_set = new_set
                    improved = True
                    break
        return current_set
    
    # Multiple strategies for ordering
    def _order_by_composite_score(moduli):
        """Order moduli by a heuristic score favoring composite numbers."""
        def score(m):
            # Count number of divisors (proxy for "composite-ness")
            divs = 0
            for i in range(2, int(m**0.5)+1):
                if m % i == 0:
                    divs += 2
                    if i*i == m:
                        divs -= 1
            return divs
        return sorted(moduli, key=score, reverse=True)
    
    def _order_by_prime_factors(moduli):
        """Order moduli by number of distinct prime factors (more = better for sharing)."""
        def omega(m):
            count = 0
            d = 2
            n = m
            while d*d <= n:
                if n % d == 0:
                    count += 1
                    while n % d == 0:
                        n //= d
                d += 1
            if n > 1:
                count += 1
            return count
        return sorted(moduli, key=omega, reverse=True)
    
    start_time = time.time()
    max_iters = 5000
    
    # Try different ordering strategies
    order_strategies = [
        lambda mods: (random.shuffle(mods), mods)[1],  # random
        _order_by_composite_score,
        _order_by_prime_factors,
        lambda mods: sorted(mods, reverse=True),  # descending
        lambda mods: sorted(mods),  # ascending
    ]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Choose a strategy (cycle through them)
        strategy_idx = it % len(order_strategies)
        moduli = all_moduli[:]
        order_strategies[strategy_idx](moduli)
        
        current_set = []
        
        for m in moduli:
            new_set = _try_add(m, current_set, gcd_table)
            if new_set is not None:
                current_set = new_set
        
        # Local search to add more
        current_set = _local_search(current_set, gcd_table, deadline)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = [list(c) for c in current_set]
    
    return best_set

def construct():
    result = _find_max_classes(max_mod=128)
    return result
# EVOLVE-BLOCK-END
