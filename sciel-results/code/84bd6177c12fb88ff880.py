"""Dense increasing sequences with unique block products."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _num_distinct_primes(n):
    """Count distinct prime factors of n."""
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def _get_all_products(seq):
    """Get all consecutive block products for a sequence."""
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products

def _check_valid(seq):
    """Check if all consecutive block products are distinct."""
    n = len(seq)
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def _greedy_build(order):
    """Build sequence greedily following the given order."""
    values = []
    products = set()
    for x in order:
        if x in values:
            continue
        # Check if adding x creates a collision
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if products.intersection(tails):
            continue
        values.append(x)
        products.update(tails)
    return values

def _local_search(values, deadline):
    """Try to improve the sequence with 1-out/1-in swaps and insertions."""
    best = list(values)
    best_len = len(best)
    
    # Precompute prime factor counts for all numbers
    pf_counts = {x: _num_distinct_primes(x) for x in range(2, 129)}
    
    # Try multiple passes of local improvement
    for iteration in range(50):
        if time.time() > deadline - 2:
            break
        
        improved = False
        
        # Try 1-out/1-in swaps
        for i, x in enumerate(best):
            for y in range(2, 129):
                if y in best:
                    continue
                if y < best[i-1] if i > 0 else False:
                    continue
                if y > best[i+1] if i < len(best)-1 else False:
                    continue
                # Try replacing x with y
                candidate = best[:i] + [y] + best[i+1:]
                if not _check_valid(candidate):
                    continue
                if len(candidate) > best_len:
                    best = candidate
                    best_len = len(candidate)
                    improved = True
                    break
            if improved:
                break
        
        # Try inserting a new element
        if not improved:
            for pos in range(len(best) + 1):
                for y in range(2, 129):
                    if y in best:
                        continue
                    if pos > 0 and y <= best[pos-1]:
                        continue
                    if pos < len(best) and y >= best[pos]:
                        continue
                    candidate = best[:pos] + [y] + best[pos:]
                    if _check_valid(candidate):
                        best = candidate
                        best_len = len(candidate)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Strategy 1: Greedy with different orderings
    best = []
    best_len = 0
    
    # Ordering by number of distinct prime factors (ascending), then by value
    order_by_pf = sorted(range(2, 129), key=lambda x: (_num_distinct_primes(x), x))
    result = _greedy_build(order_by_pf)
    if len(result) > best_len:
        best = result
        best_len = len(result)
    
    # Ordering by value (original greedy)
    order_by_val = list(range(2, 129))
    result = _greedy_build(order_by_val)
    if len(result) > best_len:
        best = result
        best_len = len(result)
    
    # Ordering: primes first, then prime powers, then composites
    def pf_key(x):
        return (_num_distinct_primes(x), x)
    order_mixed = sorted(range(2, 129), key=pf_key)
    result = _greedy_build(order_mixed)
    if len(result) > best_len:
        best = result
        best_len = len(result)
    
    # Try random orderings
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        random_order = list(range(2, 129))
        random.shuffle(random_order)
        # Bias towards numbers with fewer prime factors
        random_order.sort(key=lambda x: (_num_distinct_primes(x) + random.random(), x))
        result = _greedy_build(random_order)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Local search from best solution
    if time.time() < deadline - 5:
        best = _local_search(best, deadline)
        best_len = len(best)
    
    return best

# EVOLVE-BLOCK-END
