"""Rewritten program: maximize admissible tuple size in [0, 5000]."""
import time
import random
from math import isqrt

N = 5000

def _primes_upto(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [i for i in range(2, limit + 1) if sieve[i]]

# Precompute primes up to a reasonable bound (we need primes up to len(tuple) which is <= 5001)
_PRIMES_SMALL = _primes_upto(5001)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # safety margin
    
    # Strategy: Greedy construction with local search
    # We want the largest subset S of [0, 5000] such that for every prime p <= |S|,
    # S does not cover all residue classes mod p.
    
    # Key insight: For large p (p > sqrt(N) roughly), it's hard to cover all p residue classes
    # because we only have ~5000 elements. For small p, it's easier to cover all classes.
    
    # The seed program starts with {1} + primes > sqrt(N) and removes elements to fix violations.
    # Let's try a different approach: start with all numbers, and remove the minimum set to make it admissible.
    # This is like a hitting set / vertex cover problem.
    
    # Alternative: Build up from scratch. Start empty, add elements one by one if admissibility is maintained.
    # But that's O(N^2) which is 25M - might be too slow in Python.
    
    # Better approach: Use the structure of the problem.
    # For a tuple of size k, we need primes up to k. 
    # The constraint is: for each prime p <= k, not all p residue classes are present.
    
    # Let's try: start with all 5001 numbers [0, 1, ..., 5000].
    # For each prime p (from large to small), if all p residue classes are covered, remove one element from the least-populated class.
    # Repeat until no violations.
    
    # Actually, the seed does something similar but starts from a smaller set.
    # Let's try starting from the full set and pruning.
    
    points = set(range(0, N + 1))
    
    # We need to check primes up to len(points) which starts at 5001
    # Process primes in descending order (larger primes are harder to satisfy, so handle them first)
    # Actually, let's think about which primes are most constraining.
    # For prime p, we need |S| < p to guarantee we can't cover all classes, but |S| can be >= p and still miss a class.
    
    # Let's iteratively fix violations.
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        k = len(points)
        # Check all primes <= k
        for p in _PRIMES_SMALL:
            if p > k:
                break
            # Count residue classes
            class_counts = {}
            for v in points:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            
            if len(class_counts) == p:
                # All classes covered, need to remove at least one element from some class
                # Remove from the smallest class to minimize impact
                min_class = min(class_counts, key=lambda r: class_counts[r])
                # Remove one element from this class (prefer larger elements? or random?)
                # Remove the largest element in this class to keep smaller numbers
                candidates = [v for v in points if v % p == min_class]
                candidates.sort(reverse=True)
                points.remove(candidates[0])
                changed = True
                break
    
    return sorted(points)
