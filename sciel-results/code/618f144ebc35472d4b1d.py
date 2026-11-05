import math
import random

def construct():
    max_mod = 128
    
    def are_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    moduli = list(range(2, max_mod + 1))
    best = []
    
    # Deterministic baseline
    moduli_sorted = sorted(moduli, key=lambda m: -m.bit_count()) # proxy for complexity?
    # Actually, let's just use the existing heuristic but add random restarts.
    
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

    # 1. Original deterministic best
    moduli_sorted = sorted(moduli, key=lambda m: -num_prime_factors(m))
    chosen = []
    for m in moduli_sorted:
        placed = False
        for r in range(m):
            ok = True
            for (r2, m2) in chosen:
                if not are_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                chosen.append((r, m))
                placed = True
                break
    if len(chosen) > len(best):
        best = chosen

    # 2. Random restarts
    for _ in range(20):
        random.shuffle(moduli)
        chosen = []
        for m in moduli:
            placed = False
            for r in range(m):
                ok = True
                for (r2, m2) in chosen:
                    if not are_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    chosen.append((r, m))
                    placed = True
                    break
        if len(chosen) > len(best):
            best = chosen

    if not best:
        return [[0, 2]]
        
    return [[r, m] for r, m in best]
