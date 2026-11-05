"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction
import random

def _primes_up_to(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def construct():
    primes = _primes_up_to(1000)
    random.seed(42)
    
    L_MAX = 20000  # Cap for search, can be increased if time permits
    
    def get_initial_coverage(classes):
        cov = [0] * (L_MAX + 1)
        for p, r in classes:
            start = r if r > 0 else p
            for x in range(start, L_MAX + 1, p):
                cov[x] += 1
        return cov

    def calc_L(cov):
        L = 0
        for i in range(1, L_MAX + 1):
            if cov[i] > 0:
                L = i
            else:
                break
        return L

    def try_optimize_residues(prime_set):
        # Initialize with random residues
        classes = [[p, random.randint(0, p-1)] for p in prime_set]
        cov = get_initial_coverage(classes)
        current_L = calc_L(cov)
        
        improved = True
        while improved:
            improved = False
            # Shuffle order to avoid bias
            idxs = list(range(len(classes)))
            random.shuffle(idxs)
            for i in idxs:
                p, r = classes[i]
                best_r = r
                best_L = current_L
                # Try all residues
                for new_r in range(p):
                    if new_r == r:
                        continue
                    # Update coverage incrementally
                    # Remove old
                    old_start = r if r > 0 else p
                    for x in range(old_start, L_MAX + 1, p):
                        cov[x] -= 1
                    # Add new
                    new_start = new_r if new_r > 0 else p
                    for x in range(new_start, L_MAX + 1, p):
                        cov[x] += 1
                    new_L = calc_L(cov)
                    if new_L > best_L:
                        best_L = new_L
                        best_r = new_r
                    # Revert for next try
                    for x in range(new_start, L_MAX + 1, p):
                        cov[x] -= 1
                    for x in range(old_start, L_MAX + 1, p):
                        cov[x] += 1
                
                if best_r != r:
                    classes[i][1] = best_r
                    current_L = best_L
                    improved = True
                    # Update coverage to reflect the change
                    old_start = r if r > 0 else p
                    for x in range(old_start, L_MAX + 1, p):
                        cov[x] -= 1
                    new_start = best_r if best_r > 0 else p
                    for x in range(new_start, L_MAX + 1, p):
                        cov[x] += 1
        return classes, current_L

    # Generate candidate prime sets
    valid_sets = []
    
    # 1. Small primes
    valid_sets.append([2, 3])
    
    # 2. Medium primes
    s2 = [3, 5, 7, 11, 13, 17, 19, 23]
    cost = sum(Fraction(1, p) for p in s2)
    if cost <= 1:
        valid_sets.append(s2)
        
    # 3. Random sets
    for _ in range(100):
        p_set = set()
        cost = Fraction(0)
        while True:
            p = random.choice(primes)
            if p in p_set: continue
            c = Fraction(1, p)
            if cost + c > 1:
                break
            p_set.add(p)
            cost += c
        if len(p_set) > 0:
            valid_sets.append(list(p_set))

    # Unique sets
    unique_sets = []
    seen = set()
    for s in valid_sets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_sets.append(s)

    best_L = 0
    best_classes = []
    
    for p_set in unique_sets:
        classes, L = try_optimize_residues(p_set)
        if L > best_L:
            best_L = L
            best_classes = classes
            
    return best_classes

# EVOLVE-BLOCK-END
