"""Rewritten program with greedy + local search for maximum valid set."""

def construct():
    import time
    import random
    from itertools import combinations
    
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute prime factors for each number
    def prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    pf = {n: prime_factors(n) for n in candidates}
    
    # Check if set S is valid: for all distinct a,b,c in S, a does not divide b*c
    def is_valid(S):
        S_list = list(S)
        for i, a in enumerate(S_list):
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for k in range(j+1, len(S_list)):
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    # More efficient validity check using factor sets
    def is_valid_fast(S):
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            # For a to divide b*c, the prime factors of a must be covered by b and c together
            # We need: for each prime power p^e in a, b*c has at least p^e
            # Simplified: a divides b*c iff gcd(a, b) and a/gcd(a,b) divides c
            # i.e., a/gcd(a,b) | c
            for j in range(i+1, n):
                b = S_list[j]
                g = _gcd(a, b)
                rem = a // g
                if rem == 1:
                    continue
                # Check if rem divides any c in S_list[k] for k > j
                for k in range(j+1, n):
                    c = S_list[k]
                    if c % rem == 0:
                        return False
        return True
    
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # Count violations for a set
    def count_violations(S):
        S_list = list(S)
        n = len(S_list)
        violations = 0
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                g = _gcd(a, b)
                rem = a // g
                if rem == 1:
                    continue
                for k in range(j+1, n):
                    c = S_list[k]
                    if c % rem == 0:
                        violations += 1
        return violations
    
    # Start with all primes
    primes = [n for n in candidates if pf[n] and all(n % d != 0 for d in range(2, int(n**0.5)+1))]
    
    # Better prime check
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i+2) == 0:
                return False
            i += 6
        return True
    
    primes = [n for n in candidates if is_prime(n)]
    
    # Greedily add composites
    S = set(primes)
    
    # Try adding each composite
    composites = [n for n in candidates if not is_prime(n)]
    
    # Sort composites by "danger" - fewer distinct prime factors first, larger numbers first
    def danger_score(n):
        # Numbers that are harder to divide are safer
        # A number a is dangerous if many pairs (b,c) satisfy a|bc
        # Approximate: numbers with small prime factors are more dangerous as divisors
        f = pf[n]
        distinct = len(set(f))
        # Smaller numbers with many factors are more dangerous
        return (distinct, -n)
    
    composites_sorted = sorted(composites, key=danger_score)
    
    for c in composites_sorted:
        if time.time() > deadline - 10:
            break
        new_S = S | {c}
        if is_valid_fast(new_S):
            S = new_S
    
    # Local search: try removing one element and adding one or two others
    best_S = set(S)
    best_size = len(S)
    
    # Local search iterations
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try all single removals
        for rem in list(S):
            if time.time() > deadline - 5:
                break
            S_without = S - {rem}
            # Try adding any element not in S
            for add in candidates:
                if add in S:
                    continue
                new_S = S_without | {add}
                if len(new_S) <= best_size:
                    continue
                if is_valid_fast(new_S):
                    S = new_S
                    best_size = len(S)
                    improved = True
                    break
            if improved:
                break
        
        # Try single additions
        for add in candidates:
            if add in S or time.time() > deadline - 5:
                continue
            new_S = S | {add}
            if is_valid_fast(new_S):
                S = new_S
                best_size = len(S)
                improved = True
    
    # Multi-step local search: remove up to 2, add up to 2
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try removing 1 or 2 elements
            rem_list = list(S)
            for n_rem in [1, 2]:
                for rems in combinations(rem_list, n_rem):
                    if time.time() > deadline - 5:
                        break
                    S_without = S - set(rems)
                    # Try adding 1 or 2 elements
                    avail = [x for x in candidates if x not in S_without]
                    for n_add in [1, 2]:
                        for adds in combinations(avail, n_add):
                            new_S = S_without | set(adds)
                            if len(new_S) <= best_size:
                                continue
                            if is_valid_fast(new_S):
                                S = new_S
                                best_size = len(S)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Final polish: try to add any remaining element
    for add in candidates:
        if add in S or time.time() > deadline - 2:
            continue
        new_S = S | {add}
        if is_valid_fast(new_S):
            S = new_S
    
    return list(S)
