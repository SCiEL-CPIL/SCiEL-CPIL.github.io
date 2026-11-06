import math
import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_disjoint(r1, m1, r2, m2):
    """
    Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint.
    They are disjoint iff (r1 - r2) is not divisible by gcd(m1, m2).
    """
    g = _gcd(m1, m2)
    return (r1 - r2) % g != 0

def _prime_factors(m):
    """Distinct prime factors of m."""
    facs = set()
    d = 2
    while d * d <= m:
        if m % d == 0:
            facs.add(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        facs.add(m)
    return facs

def _find_residue(m_new, existing_classes):
    """
    Try to find a residue r in [0, m_new) that is disjoint from all existing classes.
    Returns the residue if found, else None.
    """
    for r in range(m_new):
        valid = True
        for r_ex, m_ex in existing_classes:
            if not _is_disjoint(r, m_new, r_ex, m_ex):
                valid = False
                break
        if valid:
            return r
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # All candidate moduli from 2 to 128
    moduli = list(range(2, 129))
    
    # Precompute prime factors for each modulus
    pf_cache = {}
    for m in moduli:
        pf_cache[m] = _prime_factors(m)
    
    best_set = []
    
    # Strategy 1: Sort by modulus value (ascending)
    # Strategy 2: Sort by number of distinct prime factors (descending) - favors numbers with more distinct primes
    # Strategy 3: Sort by "tightness" - fewer distinct prime factors first? Or more?
    # Inspiration 1 suggests: "Order moduli by number of distinct prime factors (descending): tightest first"
    # Let's interpret "tightest" as most constrained. Actually, the inspiration says:
    # "Order moduli by number of distinct prime factors (descending): tightest first"
    # and then fills them.
    
    # Let's try multiple orderings:
    # 1. Descending number of distinct prime factors (most complex moduli first)
    # 2. Ascending number of distinct prime factors (simplest moduli first)
    # 3. Ascending modulus value
    # 4. Descending modulus value
    # 5. Randomized permutations
    
    def omega(m):
        return len(pf_cache[m])
    
    orderings = []
    
    # Ordering 1: Descending omega
    order1 = sorted(moduli, key=lambda m: (-omega(m), m))
    orderings.append(order1)
    
    # Ordering 2: Ascending omega
    order2 = sorted(moduli, key=lambda m: (omega(m), m))
    orderings.append(order2)
    
    # Ordering 3: Ascending modulus
    order3 = sorted(moduli)
    orderings.append(order3)
    
    # Ordering 4: Descending modulus
    order4 = sorted(moduli, reverse=True)
    orderings.append(order4)
    
    # Ordering 5: Sort by number of divisors (descending)
    def num_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count
    order5 = sorted(moduli, key=lambda m: (-num_divisors(m), m))
    orderings.append(order5)
    
    # Add some random permutations
    random.seed(42)
    for _ in range(10):
        random_moduli = moduli[:]
        random.shuffle(random_moduli)
        orderings.append(random_moduli)
        
    for order in orderings:
        if time.time() > deadline:
            break
            
        current_set = []
        for m in order:
            r = _find_residue(m, current_set)
            if r is not None:
                current_set.append((r, m))
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
    # If we still have time, try a more sophisticated approach:
    # Greedy selection with backtracking or local search?
    # For now, the multiple orderings should be sufficient.
    
    # Final check to ensure we return a list of lists [residue, modulus]
    return [[r, m] for r, m in best_set]
