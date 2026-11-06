"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

MAX_MOD = 128
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _disjoint(r1, m1, r2, m2):
    """Two residue classes are disjoint iff gcd(m1,m2) does not divide (r2-r1)."""
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _greedy_build(moduli_order, time_limit):
    """Greedily assign residues for the given moduli in order."""
    pairs = []
    for m in moduli_order:
        best_r = None
        for r in range(m):
            ok = True
            for (r2, m2) in pairs:
                if not _disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                best_r = r
                break
        if best_r is not None:
            pairs.append([best_r, m])
    return pairs

def construct():
    start_time = time.time()
    
    # Strategy: try different orderings of moduli to maximize count
    # All moduli from 2 to 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Try greedy with moduli sorted in various ways
    best_pairs = []
    
    # Ordering 1: descending (larger moduli first, more residue choices)
    order1 = sorted(all_moduli, reverse=True)
    pairs1 = _greedy_build(order1, DEADLINE)
    if len(pairs1) > len(best_pairs):
        best_pairs = pairs1
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 2:
        return best_pairs
    
    # Ordering 2: ascending
    order2 = sorted(all_moduli)
    pairs2 = _greedy_build(order2, DEADLINE)
    if len(pairs2) > len(best_pairs):
        best_pairs = pairs2
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 2:
        return best_pairs
    
    # Ordering 3: descending by prime factorization complexity
    # Prefer moduli with more/smaller prime factors
    def prime_factor_score(m):
        n = m
        score = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                score += 1
                n //= d
            d += 1
        if n > 1:
            score += 1
        return score
    
    order3 = sorted(all_moduli, key=lambda x: (-prime_factor_score(x), -x))
    pairs3 = _greedy_build(order3, DEADLINE)
    if len(pairs3) > len(best_pairs):
        best_pairs = pairs3
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 2:
        return best_pairs
    
    # Try multiple random orderings
    import random
    rng = random.Random(42)
    for trial in range(50):
        if time.time() - start_time > DEADLINE - 2:
            break
        order = all_moduli[:]
        rng.shuffle(order)
        pairs = _greedy_build(order, DEADLINE)
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
    
    return best_pairs
# EVOLVE-BLOCK-END
