"""Rewritten program: maximize pairwise disjoint residue classes with distinct moduli 2..128."""
# EVOLVE-BLOCK-START
import math

def construct():
    max_mod = 128
    all_moduli = list(range(2, max_mod + 1))
    
    def are_disjoint(r1, m1, r2, m2):
        d = math.gcd(m1, m2)
        if d == 1:
            return True
        return (r1 - r2) % d != 0
    
    def try_add_residue(current, new_m):
        for r in range(new_m):
            ok = True
            for (ri, mi) in current:
                if not are_disjoint(r, new_m, ri, mi):
                    ok = False
                    break
            if ok:
                return r
        return None
    
    # Identify primes
    primes = set()
    for m in all_moduli:
        is_prime = True
        for p in range(2, int(math.isqrt(m)) + 1):
            if m % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.add(m)
    
    # Start with primes, residue 0
    current = [(0, p) for p in sorted(primes)]
    
    # Count prime factors
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    composites = [m for m in all_moduli if m not in primes]
    composites.sort(key=lambda x: (num_prime_factors(x), x))
    
    for m in composites:
        r = try_add_residue(current, m)
        if r is not None:
            current.append((r, m))
    
    result = [[r, m] for (r, m) in current]
    
    if not result:
        result = [[0, 2]]
    
    return result
# EVOLVE-BLOCK-END
