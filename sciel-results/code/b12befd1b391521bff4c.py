import time
import random
from fractions import Fraction
import numpy as np

def _sieve(n):
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def _compute_L_from_coverage(coverage, max_check=100000):
    """Compute L from coverage boolean array."""
    L = 0
    limit = min(max_check, len(coverage) - 1)
    for i in range(1, limit + 1):
        if coverage[i]:
            L = i
        else:
            break
    return L

def _build_coverage(classes, max_L=100000):
    """Build coverage boolean array from classes."""
    coverage = np.zeros(max_L + 1, dtype=np.uint8)
    for p, r in classes:
        start_k = r if r > 0 else p
        coverage[start_k::p] = 1
    return coverage

def _greedy_init(primes, max_L=100000):
    """Greedy initialization: pick primes that cover the most new integers."""
    coverage = np.zeros(max_L + 1, dtype=np.uint8)
    used_primes = set()
    classes = []
    budget = Fraction(0)
    
    # Try to cover 1..max_L efficiently
    # Start with a reasonable set of primes
    # Strategy: use primes that are "efficient" - cover many new numbers per unit budget
    
    # First, let's try a fixed good set
    # Based on the problem, primes >= 13 give sum 1/p < 1
    # Let's try different starting points
    
    return classes, coverage

def _local_search(classes, primes, max_L=100000, deadline=None, n_iter=5000):
    """Local search: try changing residues and swapping primes."""
    used_primes = set(p for p, r in classes)
    best_classes = [list(c) for c in classes]
    coverage = _build_coverage(best_classes, max_L)
    best_L = _compute_L_from_coverage(coverage, max_L)
    
    budget = sum(Fraction(1, p) for p, r in best_classes)
    remaining_budget = 1 - budget
    
    rng = random.Random(42)
    no_improve = 0
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.6:
            # Change residue of an existing prime
            if not best_classes:
                continue
            idx = rng.randrange(len(best_classes))
            p, old_r = best_classes[idx]
            new_r = rng.randrange(p)
            if new_r == old_r:
                continue
            
            # Compute new coverage
            new_coverage = coverage.copy()
            old_start = old_r if old_r > 0 else p
            new_start = new_r if new_r > 0 else p
            
            # Remove old
            new_coverage[old_start::p] = 0
            # But other classes might cover those, so we need to be careful
            # Actually, let's just recompute from scratch for correctness in this step
            # But that's expensive. Let's use a different approach.
            
            # For now, let's just try and recompute
            test_classes = [list(c) for c in best_classes]
            test_classes[idx][1] = new_r
            new_coverage = _build_coverage(test_classes, max_L)
            new_L = _compute_L_from_coverage(new_coverage, max_L)
            
            if new_L > best_L:
                best_L = new_L
                best_classes = test_classes
                coverage = new_coverage
                no_improve = 0
            else:
                no_improve += 1
                
        elif move_type < 0.8:
            # Add a new prime
            candidates = [p for p in primes if p not in used_primes]
            if not candidates:
                continue
            p = rng.choice(candidates)
            if budget + Fraction(1, p) > 1:
                continue
            new_r = rng.randrange(p)
            test_classes = [list(c) for c in best_classes] + [[p, new_r]]
            new_coverage = _build_coverage(test_classes, max_L)
            new_L = _compute_L_from_coverage(new_coverage, max_L)
            
            if new_L > best_L:
                best_L = new_L
                best_classes = test_classes
                coverage = new_coverage
                used_primes.add(p)
                budget += Fraction(1, p)
                no_improve = 0
            else:
                no_improve += 1
                
        else:
            # Remove a prime
            if len(best_classes) <= 1:
                continue
            idx = rng.randrange(len(best_classes))
            p = best_classes[idx][0]
            test_classes = [list(c) for c in best_classes if c[0] != p]
            new_coverage = _build_coverage(test_classes, max_L)
            new_L = _compute_L_from_coverage(new_coverage, max_L)
            
            if new_L > best_L:
                best_L = new_L
                best_classes = test_classes
                coverage = new_coverage
                used_primes.discard(p)
                budget -= Fraction(1, p)
                no_improve = 0
            else:
                no_improve += 1
    
    return best_L, best_classes

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    primes = _sieve(1000)
    
    # Generate multiple initial configurations
    best_L = 0
    best_classes = []
    
    # Strategy 1: Use primes starting from various points
    start_primes_options = [
        [p for p in primes if p >= 13],
        [p for p in primes if p >= 7],
        [p for p in primes if p >= 11],
        [p for p in primes if p >= 17],
        [2] + [p for p in primes if p >= 13],
        [3] + [p for p in primes if p >= 13],
        [5] + [p for p in primes if p >= 13],
        [7] + [p for p in primes if p >= 13],
    ]
    
    for primes_set in start_primes_options:
        if time.time() > deadline - 30:
            break
        
        # Check budget
        s = sum(Fraction(1, p) for p in primes_set)
        if s > 1:
            continue
        
        # Initialize with greedy residues
        classes = []
        for p in primes_set:
            # Try to cover the gap
            classes.append([p, 1])  # Start with residue 1
        
        # Run local search
        L, cls = _local_search(classes, primes, max_L=100000, deadline=deadline - 20, n_iter=2000)
        if L > best_L:
            best_L = L
            best_classes = cls
    
    # Verify budget
    s = sum(Fraction(1, p) for p, r in best_classes)
    if s > 1:
        # Fallback
        seed_set = [p for p in primes if p >= 13]
        best_classes = [[p, 1] for p in seed_set]
    
    return best_classes

if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.1f}s, Classes: {len(result)}")
    max_L = 100000
    coverage = np.zeros(max_L + 1, dtype=np.uint8)
    for p, r in result:
        start_k = r if r > 0 else p
        coverage[start_k::p] = 1
    L = 0
    for i in range(1, max_L + 1):
        if coverage[i]:
            L = i
        else:
            break
    print(f"L = {L}")
    s = sum(Fraction(1, p) for p, r in result)
    print(f"Sum 1/p = {float(s):.6f}")
