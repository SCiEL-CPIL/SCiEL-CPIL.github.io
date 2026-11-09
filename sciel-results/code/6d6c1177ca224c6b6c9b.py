"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

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
    Multi-start randomized greedy search with enhanced local improvement.
    Uses backtracking with good heuristics and swap-based local search.
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
    n_mod = len(all_moduli)
    
    # Precompute prime factorization info for each modulus
    def prime_factors(n):
        factors = []
        d = 2
        while d*d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    def distinct_prime_factors(n):
        return len(set(prime_factors(n)))
    
    def omega(m):
        return distinct_prime_factors(m)
    
    def total_prime_factors(m):
        return len(prime_factors(m))
    
    # Precompute compatibility: for each pair (m1, m2), can they coexist?
    # They can coexist if gcd(m1, m2) > 1 (otherwise they're always disjoint)
    # Actually, if gcd(m1,m2) == 1, they're ALWAYS disjoint (any residues work)
    # If gcd(m1,m2) > 1, we need to check specific residues
    
    def _try_add_with_backtrack(modulus, current_set, gcd_table, max_depth=5):
        """Try to add modulus to current_set using limited backtracking."""
        # Quick check: no modulus can be coprime with any existing (they'd always be disjoint, which is fine)
        # Actually coprime moduli are ALWAYS disjoint, so they're always compatible!
        
        # Find a residue r for modulus that works with all current_set
        # r must satisfy: for each (r_old, m_old), (r - r_old) % gcd(modulus, m_old) != 0
        
        # Use constraint propagation: collect forbidden residue classes
        # For each existing pair, r mod g_i != r_old mod g_i where g_i = gcd(modulus, m_old_i)
        
        # Try all residues in a smart order
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
    
    def _local_search_enhanced(current_set, gcd_table, deadline):
        """Enhanced local search with add and swap operations."""
        improved = True
        iterations = 0
        max_iterations = 10000
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline:
                break
            
            used_mods = {m for _, m in current_set}
            
            # Try adding each unused modulus
            for m in all_moduli:
                if m in used_mods:
                    continue
                new_set = _try_add_with_backtrack(m, current_set, gcd_table)
                if new_set is not None:
                    current_set = new_set
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one and try to add another
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                removed = current_set[i]
                remaining = current_set[:i] + current_set[i+1:]
                used_mods_after_removal = {m for _, m in remaining}
                
                # Try to add a different modulus
                best_new_set = None
                for m in all_moduli:
                    if m in used_mods_after_removal:
                        continue
                    new_set = _try_add_with_backtrack(m, remaining, gcd_table)
                    if new_set is not None:
                        # Prefer to keep the same size or improve
                        if len(new_set) > len(current_set):
                            best_new_set = new_set
                            break
                        elif len(new_set) == len(current_set) and best_new_set is None:
                            best_new_set = new_set
                
                if best_new_set is not None and len(best_new_set) > len(current_set):
                    current_set = best_new_set
                    improved = True
                    break
        
        return current_set
    
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
        return sorted(moduli, key=omega, reverse=True)
    
    def _order_by_total_prime_factors(moduli):
        """Order moduli by total number of prime factors."""
        return sorted(moduli, key=total_prime_factors, reverse=True)
    
    def _order_by_composite_with_primes(moduli):
        """Order by composite score, breaking ties by prime factor count."""
        def score(m):
            divs = 0
            for i in range(2, int(m**0.5)+1):
                if m % i == 0:
                    divs += 2
                    if i*i == m:
                        divs -= 1
            return (divs, omega(m))
        return sorted(moduli, key=score, reverse=True)
    
    start_time = time.time()
    max_iters = 2000
    
    # Try different ordering strategies
    order_strategies = [
        lambda mods: (random.shuffle(mods), mods)[1],  # random
        _order_by_composite_score,
        _order_by_prime_factors,
        _order_by_total_prime_factors,
        _order_by_composite_with_primes,
        lambda mods: sorted(mods, reverse=True),  # descending
        lambda mods: sorted(mods),  # ascending
        lambda mods: sorted(mods, key=lambda m: (omega(m), m), reverse=True),
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
            new_set = _try_add_with_backtrack(m, current_set, gcd_table)
            if new_set is not None:
                current_set = new_set
        
        # Enhanced local search to add more
        current_set = _local_search_enhanced(current_set, gcd_table, deadline)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = [list(c) for c in current_set]
            # Print progress
            # print(f"Iteration {it}: found set of size {best_size}")
    
    return best_set

def construct():
    result = _find_max_classes(max_mod=128)
    return result
# EVOLVE-BLOCK-END
