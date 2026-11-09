"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import math

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Generate primes up to 1000
    primes = _primes_upto(1000)
    
    # Start with the seed strategy (primes from 13 upward)
    seed_classes = _seed_classes(primes)
    seed_L = _compute_L(seed_classes)
    
    best_L = seed_L
    best_classes = seed_classes
    
    # Compute seed budget
    seed_primes = [p for p in primes if p >= 13]
    seed_budget = sum(Fraction(1, p) for p in seed_primes)
    
    # Try adding small primes 2, 3, 5, 7, 11 one by one with best residue
    current_classes = list(best_classes)
    current_budget = seed_budget
    
    small_primes = [2, 3, 5, 7, 11]
    
    for p in small_primes:
        cost = Fraction(1, p)
        if current_budget + cost > Fraction(1):
            continue
        
        # Try all residues for this prime
        best_residue = None
        best_new_L = _compute_L(current_classes)
        
        for r in range(p):
            test_classes = current_classes + [[p, r]]
            test_L = _compute_L(test_classes)
            if test_L > best_new_L:
                best_new_L = test_L
                best_residue = r
        
        if best_residue is not None:
            current_classes.append([p, best_residue])
            current_budget += cost
            if best_new_L > best_L:
                best_L = best_new_L
                best_classes = list(current_classes)
        
        if time.time() > deadline - 5:
            break
    
    # Try to optimize residues of existing primes to extend L.
    # Multiple passes for better convergence.
    
    if time.time() < deadline - 10:
        # Iterate multiple times
        for iteration in range(8):
            if time.time() > deadline - 5:
                break
            improved = False
            for i, (p, r) in enumerate(best_classes):
                best_r = r
                best_L_i = best_L
                for r2 in range(p):
                    if r2 == r:
                        continue
                    test = list(best_classes)
                    test[i] = [p, r2]
                    test_L = _compute_L(test)
                    if test_L > best_L_i:
                        best_L_i = test_L
                        best_r = r2
                if best_r != r:
                    best_classes[i] = [p, best_r]
                    best_L = best_L_i
                    improved = True
            if not improved:
                break
    
    # Try to add more primes from the unused pool, prioritizing those that help most
    if time.time() < deadline - 15:
        used_primes = set(p for p, r in best_classes)
        remaining_primes = [p for p in primes if p not in used_primes]
        
        # Sort by ascending size (smaller primes have higher coverage density)
        remaining_primes.sort()
        
        for p in remaining_primes:
            if time.time() > deadline - 5:
                break
            
            cost = Fraction(1, p)
            if current_budget + cost > Fraction(1):
                continue
            
            # Try all residues for this prime
            best_residue = None
            best_new_L = best_L
            
            for r in range(p):
                test_classes = best_classes + [[p, r]]
                test_L = _compute_L(test_classes)
                if test_L > best_new_L:
                    best_new_L = test_L
                    best_residue = r
            
            if best_residue is not None and best_new_L > best_L:
                best_classes.append([p, best_residue])
                current_budget += cost
                best_L = best_new_L
                used_primes.add(p)
        
        # Another round of residue optimization after adding new primes
        if time.time() < deadline - 10:
            for iteration in range(5):
                if time.time() > deadline - 5:
                    break
                improved = False
                for i, (p, r) in enumerate(best_classes):
                    best_r = r
                    best_L_i = best_L
                    for r2 in range(p):
                        if r2 == r:
                            continue
                        test = list(best_classes)
                        test[i] = [p, r2]
                        test_L = _compute_L(test)
                        if test_L > best_L_i:
                            best_L_i = test_L
                            best_r = r2
                    if best_r != r:
                        best_classes[i] = [p, best_r]
                        best_L = best_L_i
                        improved = True
                if not improved:
                    break
    
    # Final polish: try swapping out poorly-performing primes for better ones
    if time.time() < deadline - 10:
        # For each prime in the set, consider removing it and replacing with another
        used_primes = set(p for p, r in best_classes)
        
        # Try removing each prime and see if we can do better with a replacement
        for i in range(len(best_classes)):
            if time.time() > deadline - 5:
                break
            
            p_remove, r_remove = best_classes[i]
            
            # Try removing this prime
            without = list(best_classes)
            without.pop(i)
            L_without = _compute_L(without)
            
            if L_without >= best_L:
                # Removing doesn't hurt; try to add a different prime
                budget_without = current_budget - Fraction(1, p_remove)
                
                # Find best prime to add
                best_add_p = None
                best_add_r = None
                best_add_L = L_without
                
                for p_add in primes:
                    if p_add in used_primes or p_add == p_remove:
                        continue
                    cost_add = Fraction(1, p_add)
                    if budget_without + cost_add > Fraction(1):
                        continue
                    
                    for r_add in range(p_add):
                        test = without + [[p_add, r_add]]
                        test_L = _compute_L(test)
                        if test_L > best_add_L:
                            best_add_L = test_L
                            best_add_p = p_add
                            best_add_r = r_add
                
                if best_add_p is not None and best_add_L > best_L:
                    best_classes = without + [[best_add_p, best_add_r]]
                    current_budget = budget_without + Fraction(1, best_add_p)
                    best_L = best_add_L
                    used_primes = set(p for p, r in best_classes)
    
    return best_classes


def _primes_upto(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]


def _seed_classes(primes):
    seed_primes = [p for p in primes if p >= 13]
    return [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]


def _compute_L(classes):
    """Compute the largest L such that every integer 1..L is covered by some class."""
    if not classes:
        return 0
    
    # For efficiency, use a bitset or set of covered integers.
    # L is capped at 10^5, but we're mostly interested in L up to ~1000.
    max_L = 10000  # reasonable upper bound for search
    
    # Build a boolean array for coverage
    covered = [False] * (max_L + 1)
    
    for p, r in classes:
        # Cover integers n where n ≡ r (mod p), i.e., n = r, r+p, r+2p, ...
        # But n must be >= 1.
        # First integer >= 1 with n ≡ r (mod p):
        if r == 0:
            first = p
        else:
            first = r
        if first < 1:
            first += p * ((1 - first + p - 1) // p)
        
        for n in range(first, max_L + 1, p):
            covered[n] = True
    
    # Find the largest L such that covered[1..L] are all True
    L = 0
    for n in range(1, max_L + 1):
        if covered[n]:
            L = n
        else:
            break
    
    return L

# EVOLVE-BLOCK-END
