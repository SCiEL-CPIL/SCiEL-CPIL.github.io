"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def _coverage_length(classes, max_check=100000):
    if not classes:
        return 0
    covered = bytearray(max_check + 1)
    for p, r in classes:
        start = r if r > 0 else p
        if start < 1:
            start += p * ((1 - start) // p + 1)
        for x in range(start, max_check + 1, p):
            covered[x] = 1
    L = 0
    for i in range(1, max_check + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _budget_frac(classes):
    s = Fraction(0)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Seed solution
    primes_seed = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    classes = [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
    
    best_L = _coverage_length(classes)
    best_classes = [c[:] for c in classes]
    
    # Local search: try to improve by changing residues
    primes_all = _sieve(1000)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        L = _coverage_length(best_classes)
        if L >= best_L:
            best_L = L
            best_classes = [c[:] for c in classes]
        
        # Find first gap
        covered = bytearray(100001)
        for p, r in best_classes:
            start = r if r > 0 else p
            if start < 1:
                start += p * ((1 - start) // p + 1)
            for x in range(start, 100001, p):
                covered[x] = 1
        gap = 1
        for i in range(1, 100001):
            if not covered[i]:
                gap = i
                break
        else:
            gap = 100001
        
        if gap >= 100000:
            break
        
        # Try changing one existing class residue to cover gap
        improved = False
        for idx in range(len(best_classes)):
            if time.time() > deadline:
                break
            p, r = best_classes[idx]
            new_r = gap % p
            if new_r != r:
                old_r = best_classes[idx][1]
                best_classes[idx][1] = new_r
                new_L = _coverage_length(best_classes)
                if new_L > L:
                    L = new_L
                    improved = True
                    classes = [c[:] for c in best_classes]
                else:
                    best_classes[idx][1] = old_r
        
        # Try adding a new prime
        if not improved:
            used = set(p for p, r in best_classes)
            budget = _budget_frac(best_classes)
            for p in primes_all:
                if p in used:
                    continue
                if budget + Fraction(1, p) > Fraction(1):
                    continue
                new_r = gap % p
                test = best_classes + [[p, new_r]]
                new_L = _coverage_length(test)
                if new_L > L:
                    best_classes = test
                    L = new_L
                    classes = [c[:] for c in best_classes]
                    improved = True
                    break
        
        if not improved and time.time() > deadline:
            break
    
    return best_classes
# EVOLVE-BLOCK-END
