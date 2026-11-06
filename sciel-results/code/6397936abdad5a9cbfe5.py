"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

def _greedy_with_order(order, deadline):
    """Run greedy in a given order of candidate numbers."""
    values = []
    products = set()
    for x in order:
        if time.time() > deadline:
            break
        # Compute all new products if we append x
        # New products: x, x*last, x*last*second_last, ...
        tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _sort_by_num_factors(n):
    """Sort key: number of distinct prime factors (fewer first)."""
    count = 0
    d = 2
    m = n
    while d * d <= m:
        if m % d == 0:
            count += 1
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        count += 1
    return count

def _sort_by_largest_prime_factor(n):
    """Sort key: largest prime factor (larger first means 'purer' primes first)."""
    lp = 1
    d = 2
    m = n
    while d * d <= m:
        if m % d == 0:
            lp = d
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        lp = m
    return -lp  # negate so larger lp comes first

def _sort_by_smallest_prime_factor(n):
    """Sort key: smallest prime factor (larger first)."""
    spf = n
    d = 2
    while d * d <= n:
        if n % d == 0:
            spf = d
            break
        d += 1
    return -spf

def _sort_by_log(n):
    """Sort by log value (smaller first)."""
    return n

def _sort_by_composite_score(n):
    """Higher score for numbers with many small factors."""
    score = 0
    d = 2
    m = n
    while d * d <= m:
        if m % d == 0:
            cnt = 0
            while m % d == 0:
                cnt += 1
                m //= d
            score += cnt * (10 - d)  # penalize small prime factors more
        d += 1
    if m > 1:
        score += (10 - m) if m < 10 else 0
    return score  # lower score = better (add first)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    all_numbers = list(range(2, 129))
    
    # Multiple orderings to try
    orderings = [
        ("natural", sorted(all_numbers)),
        ("by_num_factors", sorted(all_numbers, key=_sort_by_num_factors)),
        ("by_lpf", sorted(all_numbers, key=_sort_by_largest_prime_factor)),
        ("by_spf", sorted(all_numbers, key=_sort_by_smallest_prime_factor)),
        ("by_composite", sorted(all_numbers, key=_sort_by_composite_score)),
        # Reverse of some
        ("rev_natural", sorted(all_numbers, reverse=True)),
        ("rev_num_factors", sorted(all_numbers, key=_sort_by_num_factors, reverse=True)),
    ]
    
    best = []
    for name, order in orderings:
        if time.time() > deadline:
            break
        result = _greedy_with_order(order, deadline)
        if len(result) > len(best):
            best = result
    
    return best
# EVOLVE-BLOCK-END

