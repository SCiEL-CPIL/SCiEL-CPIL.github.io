"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

_CAP = 100000


def _primes_upto(n):
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    import math
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def _greedy_next_gap(primes, cap, deadline, rng):
    covered = np.zeros(cap + 1, dtype=bool)
    used = set()
    budget = Fraction(1)
    classes = []
    f = 1
    order = list(primes)
    rng.shuffle(order)
    while time.time() < deadline:
        while f <= cap and covered[f]:
            f += 1
        if f > cap:
            break
        
        best_p = None
        best_r = None
        best_next_gap = 0
        best_tie_breaker = 0
        
        # Filter available primes
        available_primes = []
        for p in order:
            if p in used:
                continue
            if Fraction(1, p) > budget:
                continue
            available_primes.append(p)
            
        if not available_primes:
            break
            
        # To make it slightly more stochastic, we can shuffle available_primes?
        # Or just rely on the initial order.
        
        for p in available_primes:
            r = f % p
            # Find next gap
            next_gap = cap + 1
            k = f + 1
            # Optimization: if p is large, the gap is likely small.
            # We can limit the search depth?
            # No, we need exact L.
            while k <= cap:
                if covered[k]:
                    k += 1
                    continue
                if k % p == r:
                    k += 1
                    continue
                next_gap = k
                break
                
            # Tie breaker: prefer smaller prime (cheaper) to save budget?
            # Or larger prime?
            # Smaller prime covers more, might be better for future.
            # Let's prefer smaller p on tie.
            tie_breaker = -p
            
            if next_gap > best_next_gap or (next_gap == best_next_gap and tie_breaker > best_tie_breaker):
                best_next_gap = next_gap
                best_p = p
                best_r = r
                best_tie_breaker = tie_breaker
                
        if best_p is None:
            break
            
        # Apply
        r = best_r
        start = r if r != 0 else best_p
        covered[start::best_p] = True
        used.add(best_p)
        budget -= Fraction(1, best_p)
        classes.append([best_p, best_r])
        
        while f <= cap and covered[f]:
            f += 1
            
    L = f - 1
    return classes, L


def _true_L(classes, cap):
    if not classes:
        return 0
    # exact coverage check for 1..cap
    covered = np.zeros(cap + 1, dtype=bool)
    for p, r in classes:
        start = r if r != 0 else p
        covered[start::p] = True
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L


def construct():
    deadline = time.time() + 110.0
    primes = _primes_upto(1000)
    best_classes = None
    best_L = 0
    rng = random.Random(12345)

    # Deterministic first pass (sorted order) then randomized restarts.
    base_order = sorted(primes)
    classes, L = _greedy_next_gap(base_order, _CAP, deadline, rng)
    L = _true_L(classes, _CAP)
    if L > best_L:
        best_L = L
        best_classes = classes

    restarts = 0
    while time.time() < deadline - 5.0 and restarts < 40:
        restarts += 1
        classes, L = _greedy_next_gap(primes, _CAP, deadline, rng)
        L = _true_L(classes, _CAP)
        if L > best_L:
            best_L = L
            best_classes = classes
        if best_L >= 999:
            break

    if best_classes is None:
        # Fallback: seed-like construction
        primes_ge13 = [p for p in primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_ge13)]

    return best_classes
# EVOLVE-BLOCK-END
