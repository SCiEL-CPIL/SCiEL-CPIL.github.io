"""
Narrow admissible tuple: maximize |S| for S ⊂ [0, N] such that for every prime p ≤ |S|,
S does not cover all residue classes mod p.

Approach: Greedy construction with randomized restarts and local repair.
- We want the largest possible S. The constraint is: for each prime p ≤ |S|, there must
  be at least one residue class mod p that is NOT hit by any element of S.
- Equivalently, S must be a "narrow" set: it avoids at least one residue class for each small prime.
- Strategy: Start with all numbers [0, N]. For each prime p (in increasing order), if S covers all p residue classes,
  remove the smallest residue class (the one with fewest elements) to free it. Repeat until stable.
- But removing elements can cause larger primes to become fully covered again, so we iterate.
- To improve, we try multiple random initial deletions and pick the best.

Actually, a better structural approach:
- The constraint only applies to primes p ≤ |S|. So if we make S large, we need to satisfy more primes.
- Key insight: For a given set S, the "bottleneck" primes are those where S nearly covers all classes.
- Greedy repair: Start with S = [0, N]. While there exists a prime p ≤ |S| such that S covers all residues mod p,
  remove the least-populated residue class mod p from S. This is the seed approach.

To do better, we can:
1. Use randomized removal: instead of always removing the smallest class, try different classes and keep the best.
2. Use a smarter order: process primes in an order that minimizes damage.
3. Use local search: after the greedy phase, try swapping elements to free up more room.

Given the 330s budget, I'll implement:
- A fast greedy repair (the seed method) as a baseline.
- Multiple randomized restarts where we randomly choose which residue class to remove when there's a conflict.
- Keep the best result found within the time budget.
"""

import time
import random
from math import isqrt

N = 5000
DEADLINE = 330.0  # seconds
START_TIME = time.time()


def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]


def _check_admissible(points, primes_list):
    """Check if points is admissible: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    point_set = set(points)
    for p in primes_list:
        if p > n:
            break
        # Check if all residue classes 0..p-1 are covered
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True


def _repair_greedy(points, primes_list, remove_strategy='min'):
    """
    Repair points to be admissible.
    remove_strategy: 'min' = remove smallest class, 'max' = remove largest, 'random' = random class
    Returns repaired list.
    """
    points = list(points)
    changed = True
    max_iter = 10000
    it = 0
    while changed and it < max_iter:
        it += 1
        changed = False
        n = len(points)
        for p in primes_list:
            if p > n:
                break
            # Count residue classes
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            if len(classes) == p:
                # All classes covered; need to remove one
                if remove_strategy == 'min':
                    target_r = min(classes.keys(), key=lambda r: len(classes[r]))
                elif remove_strategy == 'max':
                    target_r = max(classes.keys(), key=lambda r: len(classes[r]))
                elif remove_strategy == 'random':
                    target_r = random.choice(list(classes.keys()))
                else:
                    target_r = min(classes.keys(), key=lambda r: len(classes[r]))
                # Remove all elements in that class
                to_remove = set(classes[target_r])
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points


def _greedy_construct(strategy='min'):
    """
    Construct an admissible tuple using greedy repair.
    """
    # Start with a good initial set: include 1 and all primes > sqrt(N)
    # Actually, let's start with all numbers and repair.
    # But that's O(N^2). Let's start with a subset.
    
    # Better initial set: start with all numbers, but that's 5001 elements.
    # The repair loop will remove elements. Let's just do it.
    
    points = list(range(N + 1))
    primes_list = _primes(N)
    
    points = _repair_greedy(points, primes_list, remove_strategy=strategy)
    return points


def _local_search(points, primes_list, time_limit=10.0):
    """
    Try to improve points by local swaps.
    For each element in points, try removing it and adding a different element.
    Keep the best.
    """
    start = time.time()
    best = list(points)
    best_len = len(best)
    point_set = set(best)
    
    # Precompute primes up to N
    # Try to add elements that don't violate admissibility
    # Actually, we want to maximize size. Try adding elements one by one.
    
    # First, try to add any element from [0, N] not in best
    candidates = [v for v in range(N + 1) if v not in point_set]
    random.shuffle(candidates)
    
    for v in candidates:
        if time.time() - start > time_limit:
            break
        test = best + [v]
        if _check_admissible(test, primes_list):
            best = test
            point_set.add(v)
            best_len += 1
    
    return best


def construct():
    """
    Main entry point. Returns a list of distinct integers in [0, N] forming an admissible tuple.
    """
    start = time.time()
    primes_list = _primes(N)
    
    best = []
    best_len = 0
    
    # First, run the standard greedy repair
    points = _greedy_construct(strategy='min')
    if len(points) > best_len:
        best = list(points)
        best_len = len(points)
    
    elapsed = time.time() - start
    if elapsed > DEADLINE - 30:
        return best
    
    # Try randomized restarts
    random.seed(42)
    for trial in range(200):
        if time.time() - start > DEADLINE - 20:
            break
        
        # Random strategy: random class removal
        points = _greedy_construct(strategy='random')
        if len(points) > best_len:
            best = list(points)
            best_len = len(points)
        
        # Try 'max' strategy occasionally
        if trial % 3 == 0:
            points = _greedy_construct(strategy='max')
            if len(points) > best_len:
                best = list(points)
                best_len = len(points)
    
    # Local search on the best
    if time.time() - start < DEADLINE - 15:
        best = _local_search(best, primes_list, time_limit=10.0)
    
    return best


# Precompute primes at module level for speed
_PRIMES = _primes(N)


def _repair_greedy_fast(points, primes_list, remove_strategy='min'):
    """Faster repair using sets."""
    points = list(points)
    point_set = set(points)
    changed = True
    max_iter = 5000
    it = 0
    while changed and it < max_iter:
        it += 1
        changed = False
        n = len(points)
        for p in primes_list:
            if p > n:
                break
            # Count residue classes efficiently
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            # Find if all classes are covered
            if min(counts) > 0:
                # All classes covered
                if remove_strategy == 'min':
                    target_r = counts.index(min(counts))
                elif remove_strategy == 'max':
                    target_r = counts.index(max(counts))
                elif remove_strategy == 'random':
                    target_r = random.randint(0, p - 1)
                else:
                    target_r = counts.index(min(counts))
                # Remove all elements with residue target_r mod p
                to_remove = [v for v in points if v % p == target_r]
                for v in to_remove:
                    point_set.discard(v)
                points = [v for v in points if v % p != target_r]
                changed = True
                break
    return points


def _greedy_construct_fast(strategy='min'):
    points = list(range(N + 1))
    points = _repair_greedy_fast(points, _PRIMES, remove_strategy=strategy)
    return points


def construct():
    start = time.time()
    best = []
    best_len = 0
    
    # Standard greedy
    points = _greedy_construct_fast(strategy='min')
    if len(points) > best_len:
        best = list(points)
        best_len = len(points)
    
    elapsed = time.time() - start
    if elapsed > DEADLINE - 30:
        return best
    
    # Randomized restarts
    random.seed(12345)
    for trial in range(500):
        if time.time() - start > DEADLINE - 25:
            break
        
        if trial % 3 == 0:
            strat = 'min'
        elif trial % 3 == 1:
            strat = 'max'
        else:
            strat = 'random'
        
        points = _greedy_construct_fast(strategy=strat)
        if len(points) > best_len:
            best = list(points)
            best_len = len(points)
    
    return best
