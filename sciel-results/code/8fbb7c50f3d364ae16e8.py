"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from fractions import Fraction

_CAP = 100000
_DEADLINE = 115  # Safety margin before 120s evaluator kill


def _sieve_primes(n):
    """Generate all primes up to n using sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def _compute_L(coverage):
    """Find the largest L such that all integers 1..L are covered."""
    # coverage is a numpy bool array where coverage[i] means i is covered
    # Find first gap
    for i in range(1, len(coverage)):
        if not coverage[i]:
            return i - 1
    return len(coverage) - 1


def _build_coverage(classes, cap=_CAP):
    """Build coverage array from list of [p, r] classes."""
    coverage = np.zeros(cap + 1, dtype=bool)
    for p, r in classes:
        # Mark all numbers n where n ≡ r (mod p)
        start = r
        if start < 1:
            start = r + p  # First positive number ≡ r mod p
        # Actually, let's be more careful
        # We want n >= 1 such that n % p == r
        # The smallest such n: if r == 0, then n = p; if r > 0, then n = r (if r >= 1)
        if r == 0:
            start = p
        else:
            start = r
        if start >= 1:
            coverage[start::p] = True
    return coverage


def _try_local_search(classes, primes, coverage, L, deadline, rng):
    """Try to improve coverage by changing remainders or swapping primes."""
    best_classes = classes[:]
    best_L = L
    best_coverage = coverage.copy()
    
    start_time = time.time()
    
    # Try changing remainders of existing classes
    for _ in range(500):  # Max iterations
        if time.time() > deadline:
            break
        
        # Randomly pick a class to modify
        idx = rng.randint(0, len(best_classes) - 1)
        p, r = best_classes[idx]
        
        # Try a random new remainder
        new_r = rng.randint(0, p - 1)
        if new_r == r:
            continue
        
        # Compute new coverage
        old_coverage = best_coverage.copy()
        best_classes[idx] = [p, new_r]
        new_coverage = _build_coverage(best_classes, _CAP)
        new_L = _compute_L(new_coverage)
        
        if new_L > best_L:
            best_L = new_L
            best_coverage = new_coverage
        else:
            # Revert
            best_classes[idx] = [p, r]
            best_coverage = old_coverage
    
    # Try swapping primes
    used_primes = set(p for p, r in best_classes)
    unused_primes = [p for p in primes if p not in used_primes]
    
    for _ in range(200):  # Max swap attempts
        if time.time() > deadline:
            break
        
        if not unused_primes:
            break
        
        # Pick a random used prime to potentially remove
        idx = rng.randint(0, len(best_classes) - 1)
        old_p, old_r = best_classes[idx]
        
        # Pick a random unused prime to add
        new_p = unused_primes[rng.randint(0, len(unused_primes) - 1)]
        
        # Check budget: we need sum(1/p) <= 1
        # Current budget usage
        current_budget = sum(Fraction(1, p) for p, r in best_classes)
        
        # If we swap, new budget = current - 1/old_p + 1/new_p
        new_budget = current_budget - Fraction(1, old_p) + Fraction(1, new_p)
        
        if new_budget > Fraction(1, 1):
            continue
        
        # Try the swap
        best_classes[idx] = [new_p, 0]  # Start with remainder 0
        new_coverage = _build_coverage(best_classes, _CAP)
        new_L = _compute_L(new_coverage)
        
        if new_L > best_L:
            best_L = new_L
            best_coverage = new_coverage
            used_primes.discard(old_p)
            used_primes.add(new_p)
            unused_primes.remove(new_p)
        else:
            best_classes[idx] = [old_p, old_r]
    
    return best_classes, best_L, best_coverage


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    primes = _sieve_primes(1000)
    rng = random.Random(42)
    
    # Initial construction: use primes from 13 upward
    # This gives sum of reciprocals about 0.92
    initial_primes = [p for p in primes if p >= 13]
    
    # Start with a reasonable initial set
    # Use greedy: add primes that cover the most new numbers
    classes = []
    coverage = np.zeros(_CAP + 1, dtype=bool)
    budget = Fraction(0, 1)
    
    # First, add primes in order of smallest (to get good coverage early)
    for p in initial_primes[:30]:  # Start with first 30 primes >= 13
        if budget + Fraction(1, p) > Fraction(1, 1):
            continue
        
        # Try all remainders and pick the one that maximizes coverage
        best_r = 0
        best_L = -1
        for r in range(p):
            # Temporarily add this class
            test_classes = classes + [[p, r]]
            test_coverage = _build_coverage(test_classes, _CAP)
            test_L = _compute_L(test_coverage)
            if test_L > best_L:
                best_L = test_L
                best_r = r
        
        classes.append([p, best_r])
        budget += Fraction(1, p)
        coverage = _build_coverage(classes, _CAP)
    
    # Compute initial L
    L = _compute_L(coverage)
    
    # Local search to improve
    while time.time() < deadline - 5:  # Leave 5s safety margin
        classes, L, coverage = _try_local_search(classes, primes, coverage, L, deadline - 2, rng)
        
        # Check if we can add more primes
        used_primes = set(p for p, r in classes)
        unused_primes = [p for p in primes if p not in used_primes]
        
        if not unused_primes:
            break
        
        # Try adding a new prime
        added = False
        for new_p in sorted(unused_primes, key=lambda x: -x):  # Try larger primes first (cheaper)
            if budget + Fraction(1, new_p) > Fraction(1, 1):
                continue
            
            # Try a few random remainders
            for r in [0, rng.randint(0, new_p - 1), rng.randint(0, new_p - 1)]:
                test_classes = classes + [[new_p, r]]
                test_coverage = _build_coverage(test_classes, _CAP)
                test_L = _compute_L(test_coverage)
                
                if test_L > L:
                    classes = test_classes
                    L = test_L
                    coverage = test_coverage
                    budget += Fraction(1, new_p)
                    added = True
                    break
            
            if added:
                break
        
        if not added and time.time() > deadline - 10:
            break
    
    return classes
# EVOLVE-BLOCK-END
