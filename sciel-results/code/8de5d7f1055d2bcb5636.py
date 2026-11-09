import math
import random
import time

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # Generate primes up to 128
    primes = []
    for i in range(2, 129):
        is_prime = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    composites = [m for m in range(2, 129) if m not in primes]
    
    # Sort composites by number of prime factors (ascending)
    # This tends to add "simpler" composites first
    def count_factors(m):
        count = 0
        temp = m
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                count += 1
                temp //= p
        if temp > 1:
            count += 1
        return count
        
    composites.sort(key=count_factors)
    
    def check_compatibility(r, m, existing):
        for r_i, m_i in existing:
            g = math.gcd(m, m_i)
            if g > 1:
                if r % g == r_i % g:
                    return False
        return True

    def greedy_search(order):
        result = []
        for m in order:
            # Try residues in random order to avoid bias
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if check_compatibility(r, m, result):
                    result.append((r, m))
                    break
        return result

    best_result = []
    
    # Deterministic attempts
    orders = [
        primes + composites,
        primes + list(reversed(composites)),
        list(reversed(primes)) + composites,
    ]
    
    for order in orders:
        res = greedy_search(order)
        if len(res) > len(best_result):
            best_result = res
            
    # Randomized restarts
    while time.time() < deadline:
        # Mix primes and composites randomly?
        # Or just random order of all moduli
        order = list(range(2, 129))
        random.shuffle(order)
        res = greedy_search(order)
        if len(res) > len(best_result):
            best_result = res
            
        # Occasionally try primes first with random composites order
        if random.random() < 0.5:
            order = primes + composites[:]
            random.shuffle(order[len(primes):]) # Shuffle composites part
            res = greedy_search(order)
            if len(res) > len(best_result):
                best_result = res

    return [[r, m] for r, m in best_result]
