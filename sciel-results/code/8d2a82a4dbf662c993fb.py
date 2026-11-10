"""Improved program for prime residue class covering task."""
import time
import math
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
import math

def _sieve_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _is_covered(n, classes):
    """Check if n is covered by any class [p, r]."""
    for p, r in classes:
        if n % p == r:
            return True
    return False

def _find_coverage_length(classes, limit=100000):
    """Find the largest L such that all 1..L are covered, up to limit."""
    if not classes:
        return 0
    for n in range(1, limit + 1):
        if not _is_covered(n, classes):
            return n - 1
    return limit

def _sum_reciprocals(classes):
    """Compute sum of 1/p for all classes using exact rationals."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def _valid_solution(classes, primes_max=1000):
    """Check if the solution is valid: distinct primes <= 1000, 0 <= r < p, sum 1/p <= 1."""
    seen_primes = set()
    for p, r in classes:
        if p > primes_max:
            return False
        if p in seen_primes:
            return False
        seen_primes.add(p)
        if not (0 <= r < p):
            return False
    if _sum_reciprocals(classes) > 1:
        return False
    return True

def _construct_greedy(primes, target_l=100000):
    """
    Greedy construction: start with empty set, add classes that maximize coverage.
    """
    classes = []
    used_primes = set()
    
    # Precompute primes
    all_primes = [p for p in primes if p <= 1000]
    
    # Start with a base covering of small numbers
    # Use primes 2, 3, 5, 7, 11 to cover initial range
    base_primes = [2, 3, 5, 7, 11]
    for p in base_primes:
        classes.append([p, 0])
        used_primes.add(p)
    
    current_l = _find_coverage_length(classes, 2000)
    
    # Try to extend coverage by adding more primes
    remaining_primes = [p for p in all_primes if p not in used_primes]
    
    max_iterations = 50
    for _ in range(max_iterations):
        best_class = None
        best_l = current_l
        
        for p in remaining_primes:
            if p in used_primes:
                continue
            # Try different residues for this prime
            # Heuristic: try residue that covers the first uncovered number
            if current_l < target_l:
                target = current_l + 1
                r = target % p
            else:
                r = 0
            
            new_classes = classes + [[p, r]]
            if not _valid_solution(new_classes):
                continue
            
            # Check how much coverage we gain
            new_l = _find_coverage_length(new_classes, min(current_l + 500, target_l))
            if new_l > best_l:
                best_l = new_l
                best_class = [p, r]
        
        if best_class is None:
            break
        
        classes.append(best_class)
        used_primes.add(best_class[0])
        current_l = best_l
        
        if current_l >= target_l:
            break
    
    return classes

def _construct_optimized():
    """
    Optimized construction using a more systematic approach.
    """
    primes = _sieve_primes(1000)
    
    # Strategy: Use small primes with carefully chosen residues
    # to maximize the covered prefix length
    
    # Start with primes 2, 3, 5, 7, 11, 13
    # For each, choose residue to cover as many consecutive integers as possible
    
    classes = []
    
    # Phase 1: Add small primes with residue 0 (covers multiples)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    for p in small_primes:
        classes.append([p, 0])
    
    current_l = _find_coverage_length(classes, 5000)
    
    # Phase 2: Try to extend by adding more primes with optimized residues
    used_primes = set(p for p, r in classes)
    remaining = [p for p in primes if p not in used_primes]
    
    for _ in range(30):
        best_class = None
        best_l = current_l
        
        for p in remaining:
            # Try residue that covers current_l + 1
            if current_l < 100000:
                target = current_l + 1
                r = target % p
            else:
                r = 0
            
            new_classes = classes + [[p, r]]
            if not _valid_solution(new_classes):
                continue
            
            new_l = _find_coverage_length(new_classes, min(current_l + 1000, 100000))
            if new_l > best_l:
                best_l = new_l
                best_class = [p, r]
        
        if best_class is None:
            break
        
        classes.append(best_class)
        used_primes.add(best_class[0])
        remaining.remove(best_class[0])
        current_l = best_l
        
        if current_l >= 100000:
            break
    
    return classes

def _construct_crt_based():
    """
    Construction based on Chinese Remainder Theorem ideas.
    Aim to cover 1..L by ensuring for each n, there's a class [p, r] with n ≡ r (mod p).
    """
    primes = _sieve_primes(1000)
    
    # We want to maximize L such that every n in 1..L is covered.
    # A key insight: if we use primes p1, p2, ..., pk with residues r1, r2, ..., rk,
    # then n is covered if n ≡ ri (mod pi) for some i.
    
    # Strategy: Use a set of primes and assign residues to maximize prefix coverage.
    # This is related to covering systems.
    
    # Start with a known covering system for small L, then extend.
    
    # Base: [2,0], [3,1], [5,2] covers 1,2,3,4,5,6,7,8,9,10
    # Let's build up from there.
    
    classes = [[2, 0], [3, 1], [5, 2], [7, 3], [11, 5]]
    
    current_l = _find_coverage_length(classes, 10000)
    
    used_primes = set(p for p, r in classes)
    remaining = [p for p in primes if p not in used_primes]
    
    for _ in range(40):
        best_class = None
        best_l = current_l
        
        for p in remaining:
            # Try several residues
            for r in range(min(p, 20)):
                new_classes = classes + [[p, r]]
                if not _valid_solution(new_classes):
                    continue
                
                new_l = _find_coverage_length(new_classes, min(current_l + 500, 100000))
                if new_l > best_l:
                    best_l = new_l
                    best_class = [p, r]
        
        if best_class is None:
            break
        
        classes.append(best_class)
        used_primes.add(best_class[0])
        remaining.remove(best_class[0])
        current_l = best_l
        
        if current_l >= 100000:
            break
    
    return classes

def construct():
    """
    Main entry point. Returns a list of [p, r] pairs.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try multiple construction strategies and pick the best
    best_classes = None
    best_l = 0
    
    strategies = [
        _construct_optimized,
        _construct_crt_based,
        _construct_greedy,
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        try:
            classes = strategy()
            if _valid_solution(classes):
                l = _find_coverage_length(classes, 100000)
                if l > best_l:
                    best_l = l
                    best_classes = classes
        except Exception:
            continue
    
    if best_classes is None:
        # Fallback: simple construction
        primes = _sieve_primes(1000)
        best_classes = [[p, 0] for p in primes[:10]]
    
    return best_classes

# EVOLVE-BLOCK-END
