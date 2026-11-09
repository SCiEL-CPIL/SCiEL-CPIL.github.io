import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_valid_subset(subset):
    """Check if the subset contains no bad triples."""
    n = len(subset)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = subset[i], subset[j], subset[k]
                if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
                    return False
    return True

def _greedy_construct(order):
    """
    Greedily add numbers in the given order if they don't form a bad triple.
    """
    selected = []
    # Precompute lcms for speed? No, N is small.
    for x in order:
        # Check against all pairs in selected
        ok = True
        for i in range(len(selected)):
            a = selected[i]
            la = _lcm(a, x)
            for j in range(i + 1, len(selected)):
                b = selected[j]
                if la == _lcm(a, b) == _lcm(b, x):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.append(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    numbers = list(range(1, 129))
    
    # Strategy 1: Primes and Prime Powers first
    def is_prime_power(n):
        if n < 2:
            return False
        p = n
        while p > 1:
            if p * p > n:
                break
            if n % p == 0:
                n //= p
            else:
                return False
        return True

    # Identify primes
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in numbers if is_prime(x)]
    prime_powers = [x for x in numbers if is_prime_power(x) and x not in primes]
    others = [x for x in numbers if x not in primes and x not in prime_powers]
    
    # Order 1: Primes, Prime Powers, Others (sorted by value)
    order1 = primes + prime_powers + others
    result1 = _greedy_construct(order1)
    
    # Order 2: Standard 1..128
    order2 = numbers
    result2 = _greedy_construct(order2)
    
    # Order 3: Sorted by number of divisors (ascending)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    order3 = sorted(numbers, key=num_divisors)
    result3 = _greedy_construct(order3)
    
    best_result = max([result1, result2, result3], key=len)
    
    # Try some random shuffles if time permits
    random.seed(42)
    for i in range(50):
        if time.time() > deadline:
            break
        random.shuffle(numbers)
        res = _greedy_construct(numbers)
        if len(res) > len(best_result):
            best_result = res
            
    # Final verification (optional but good for safety, though slow for large sets)
    # For N=128, _is_valid_subset is O(N^3) ~ 2M ops, which is fast enough.
    # But we trust the greedy check.
    
    return best_result
