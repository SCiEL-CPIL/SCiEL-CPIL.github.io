import time
from math import gcd
from math import lcm as math_lcm

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin
    
    def is_bad_triple(a, b, c):
        return math_lcm(a, b) == math_lcm(a, c) == math_lcm(b, c)
    
    def can_add(x, S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                if is_bad_triple(a, b, x):
                    return False
        return True
    
    # Step 1: Build base set of all prime powers
    def is_prime_power(n):
        if n == 1:
            return True
        # Check if n = p^k for some prime p and k >= 1
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                m = n
                while m % p == 0:
                    m //= p
                if m == 1:
                    return True
                return False
        return True  # n is prime
    
    prime_powers = set()
    for x in range(1, 129):
        if is_prime_power(x):
            prime_powers.add(x)
    
    selected = set(prime_powers)
    
    # Step 2: Try to add numbers with 3+ distinct prime factors
    def num_distinct_primes(n):
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
    
    candidates = [x for x in range(1, 129) if x not in selected]
    # Sort by number of distinct prime factors descending, then by value
    candidates.sort(key=lambda x: (-num_distinct_primes(x), x))
    
    for x in candidates:
        if time.time() - start_time > deadline:
            break
        if can_add(x, selected):
            selected.add(x)
    
    # Step 3: Local search - try to add remaining
    remaining = [x for x in range(1, 129) if x not in selected]
    for x in remaining:
        if time.time() - start_time > deadline:
            break
        if can_add(x, selected):
            selected.add(x)
    
    # Step 4: Local search with swaps
    # Try removing one element and adding two
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > deadline:
            break
        S_list = list(selected)
        for i in range(len(S_list)):
            if time.time() - start_time > deadline:
                break
            a = S_list[i]
            # Try removing a and adding two from remaining
            rem = [x for x in range(1, 129) if x not in selected]
            for b in rem:
                if time.time() - start_time > deadline:
                    break
                S2 = selected - {a}
                if can_add(b, S2):
                    S2.add(b)
                    for c in rem:
                        if c == b:
                            continue
                        if time.time() - start_time > deadline:
                            break
                        if can_add(c, S2):
                            selected = S2 | {c}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return list(selected)
