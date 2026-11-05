import math
from itertools import combinations

def construct():
    max_mod = 128
    # Try to find maximum pairwise disjoint residue classes
    # Two classes (r1,m1) and (r2,m2) are disjoint iff r1 != r2 mod gcd(m1,m2)
    
    def are_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    # Greedy: try each modulus, pick a residue that's compatible with all chosen
    # Order moduli by some heuristic
    moduli = list(range(2, max_mod + 1))
    
    # Try multiple orderings and take the best
    best = []
    
    # Strategy: for each modulus, try all residues, pick one that works
    # Sort moduli by number of prime factors (more factors = more constraints)
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
    
    # Try ordering by decreasing number of prime factors (more constrained first)
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
        if not placed:
            continue
    
    best = [[r, m] for r, m in chosen]
    
    # Also try a different ordering: increasing modulus
    moduli_sorted2 = sorted(moduli)
    chosen2 = []
    for m in moduli_sorted2:
        placed = False
        for r in range(m):
            ok = True
            for (r2, m2) in chosen2:
                if not are_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                chosen2.append((r, m))
                placed = True
                break
    
    if len(chosen2) > len(best):
        best = [[r, m] for r, m in chosen2]
    
    # Try yet another: prioritize prime powers
    prime_powers = []
    for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127]:
        pp = p
        while pp * p <= max_mod:
            prime_powers.append(pp * p)
            pp *= p
        prime_powers.append(p)
    prime_powers = list(set(pp for pp in prime_powers if 2 <= pp <= max_mod))
    prime_powers.sort()
    
    chosen3 = []
    for m in prime_powers:
        placed = False
        for r in range(m):
            ok = True
            for (r2, m2) in chosen3:
                if not are_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                chosen3.append((r, m))
                placed = True
                break
    
    if len(chosen3) > len(best):
        best = [[r, m] for r, m in chosen3]
    
    if not best:
        return [[0, 2]]
    
    return best
