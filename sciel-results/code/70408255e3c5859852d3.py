"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _get_primes(limit=1000):
    """Return list of primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def _compute_coverage(classes, cap=100000):
    """Compute the largest L such that every integer 1..L is covered."""
    if not classes:
        return 0
    covered = [False] * (cap + 2)
    for p, r in classes:
        start = r if r >= 1 else p
        for n in range(start, cap + 1, p):
            covered[n] = True
    L = 0
    for n in range(1, cap + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _sum_reciprocals(classes):
    """Compute sum of 1/p as Fraction."""
    s = Fraction(0)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def construct():
    deadline = time.time() + 115  # 115s safety margin for 120s limit
    primes = _get_primes(1000)
    
    # Start with a greedy construction: pick primes to cover consecutive integers
    # Use a strategy that builds coverage from 1 upward
    best_classes = []
    best_L = 0
    best_sum = Fraction(0)
    
    # Greedy: for each next uncovered integer, find a prime/residue that covers it
    # and extends coverage the most
    used_primes = set()
    
    # First, build an initial solution greedily
    for _ in range(50):  # try to add up to 50 classes
        if time.time() > deadline:
            break
        if best_sum >= Fraction(1):
            break
        
        # Find the first uncovered integer
        # We need to check coverage efficiently
        # For small L, just check directly
        next_uncovered = best_L + 1
        # Check if it's actually uncovered
        covered = False
        for p, r in best_classes:
            if next_uncovered % p == r:
                covered = True
                break
        
        if covered:
            # Find next uncovered
            n = best_L + 1
            while n <= 100000:
                found = False
                for p, r in best_classes:
                    if n % p == r:
                        found = True
                        break
                if found:
                    n += 1
                else:
                    break
            next_uncovered = n
            if n > 100000:
                break
        
        # Try to find a prime that covers next_uncovered and extends coverage
        best_new_L = best_L
        best_new_class = None
        best_new_sum = best_sum
        
        for p in primes:
            if p in used_primes:
                continue
            new_sum = best_sum + Fraction(1, p)
            if new_sum > Fraction(1):
                continue
            
            # The residue must satisfy next_uncovered % p == r
            r = next_uncovered % p
            
            # Try this class
            new_classes = best_classes + [[p, r]]
            new_L = _compute_coverage(new_classes)
            if new_L > best_new_L:
                best_new_L = new_L
                best_new_class = [p, r]
                best_new_sum = new_sum
        
        if best_new_class is not None and best_new_L > best_L:
            best_classes.append(best_new_class)
            used_primes.add(best_new_class[0])
            best_L = best_new_L
            best_sum = best_new_sum
        else:
            break
    
    # Now do local search to improve
    # Randomized local search: try random perturbations and accept improvements
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Randomly perturb one class
        if len(best_classes) == 0:
            break
        
        idx = random.randint(0, len(best_classes) - 1)
        old_p, old_r = best_classes[idx]
        
        # Choose a random new prime and residue
        if random.random() < 0.5:
            # Change residue only
            new_p = old_p
            new_r = random.randint(0, old_p - 1)
            new_sum = best_sum
        else:
            # Change prime and residue
            used = set(p for p, r in best_classes)
            available = [p for p in primes if p not in used]
            if not available:
                available = [p for p in primes if p != old_p]
            if not available:
                continue
            new_p = random.choice(available)
            new_r = random.randint(0, new_p - 1)
            new_sum = best_sum - Fraction(1, old_p) + Fraction(1, new_p)
            if new_sum > Fraction(1):
                continue
        
        new_classes = best_classes[:]
        new_classes[idx] = [new_p, new_r]
        new_L = _compute_coverage(new_classes)
        
        if new_L > best_L:
            best_L = new_L
            best_classes = new_classes
            best_sum = new_sum
    
    # Try to add more classes if budget allows
    for _ in range(100):
        if time.time() > deadline:
            break
        if best_sum >= Fraction(1):
            break
        
        # Find first uncovered
        n = best_L + 1
        found_uncovered = False
        while n <= 100000:
            covered = False
            for p, r in best_classes:
                if n % p == r:
                    covered = True
                    break
            if covered:
                n += 1
            else:
                found_uncovered = True
                break
        
        if not found_uncovered:
            break
        
        # Try to add a class covering n
        used_primes = set(p for p, r in best_classes)
        best_new_L = best_L
        best_new_class = None
        best_new_sum = best_sum
        
        for p in primes:
            if p in used_primes:
                continue
            new_sum = best_sum + Fraction(1, p)
            if new_sum > Fraction(1):
                continue
            r = n % p
            new_classes = best_classes + [[p, r]]
            new_L = _compute_coverage(new_classes)
            if new_L > best_new_L:
                best_new_L = new_L
                best_new_class = [p, r]
                best_new_sum = new_sum
        
        if best_new_class is not None and best_new_L > best_L:
            best_classes.append(best_new_class)
            best_L = best_new_L
            best_sum = best_new_sum
        else:
            break
    
    return best_classes
# EVOLVE-BLOCK-END
