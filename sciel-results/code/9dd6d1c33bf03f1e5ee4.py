import time
import random
from math import gcd

DEADLINE = 50.0

# Precompute prime factorizations for numbers 2..256
def _prime_factorize(n):
    """Return dict of prime factors with exponents"""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

# Precompute for all numbers in universe
_FACTORS = {}
for _n in range(2, 257):
    _FACTORS[_n] = _prime_factorize(_n)

def _divides(a, b):
    """Check if a divides b using precomputed factors"""
    # a | b iff for every prime p, exponent of p in a <= exponent of p in b
    for p, ea in _FACTORS[a].items():
        eb = _FACTORS[b].get(p, 0)
        if ea > eb:
            return False
    return True

def _divides_product(a, b, c):
    """Check if a divides b*c using precomputed factors"""
    # a | b*c iff for every prime p, exponent of p in a <= exponent of p in b + exponent of p in c
    for p, ea in _FACTORS[a].items():
        eb = _FACTORS[b].get(p, 0)
        ec = _FACTORS[c].get(p, 0)
        if ea > eb + ec:
            return False
    return True

def _valid_add(C, a):
    """Check if a can be added to set C without violating constraints"""
    n = len(C)
    for i in range(n):
        b = C[i]
        for j in range(i + 1, n):
            c = C[j]
            if _divides_product(a, b, c):
                return False
            if _divides_product(b, a, c):
                return False
            if _divides_product(c, a, b):
                return False
    return True

def _is_valid_set(S):
    """Check if S is a valid set"""
    C = list(S)
    n = len(C)
    for i in range(n):
        a = C[i]
        for j in range(i + 1, n):
            b = C[j]
            for k in range(j + 1, n):
                c = C[k]
                if _divides_product(a, b, c) or _divides_product(b, a, c) or _divides_product(c, a, b):
                    return False
    return True

def construct():
    t0 = time.time()
    universe = list(range(2, 257))
    
    def greedy_construct(order_key, seed=0):
        random.seed(seed)
        C = []
        # Create a list of (key, value) and sort
        # If keys are equal, randomize
        items = [(order_key(x), random.random(), x) for x in universe]
        items.sort()
        for _, _, x in items:
            if time.time() - t0 > DEADLINE:
                break
            if _valid_add(C, x):
                C.append(x)
        return C

    def local_search(S, max_iters=100):
        # S is a list
        S_set = set(S)
        improved = True
        while improved and (time.time() - t0 < DEADLINE - 5.0):
            improved = False
            # Try to add any missing element
            for x in universe:
                if x in S_set:
                    continue
                if _valid_add(S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break # Restart loop with new S
            
            if not improved:
                # Try swaps: remove x, add y
                # To keep it fast, only try swaps if we are stuck
                # Find a y not in S that conflicts with some x in S
                # Try swapping x and y
                # This is O(N^2 * ValidityCheck) which might be slow.
                # Limit swaps to a few random attempts or specific candidates.
                
                # Heuristic: Pick a random y not in S.
                # Find conflicting x's.
                # Try removing one and adding y.
                
                candidates = [x for x in universe if x not in S_set]
                random.shuffle(candidates)
                
                for y in candidates[:10]: # Limit attempts
                    if time.time() - t0 > DEADLINE - 5.0:
                        break
                    # Find conflicts of y with S
                    conflicts = []
                    for i in range(len(S)):
                        b = S[i]
                        for j in range(i + 1, len(S)):
                            c = S[j]
                            if (b * c) % y == 0 or (y * c) % b == 0 or (y * b) % c == 0:
                                # Conflict found involving b, c, y
                                # Any of b, c could be removed?
                                # Actually, if we remove b, the triple (y, b, c) is gone.
                                # But y might conflict with other pairs.
                                # We need to check if S \ {b} U {y} is valid.
                                conflicts.append(b)
                                conflicts.append(c)
                    
                    for x in set(conflicts):
                        # Try swap x -> y
                        S_new = [z for z in S if z != x]
                        if _valid_add(S_new, y):
                            # Check if the whole set S_new U {y} is valid
                            # _valid_add checks y against S_new.
                            # Since S was valid, S_new is valid.
                            # So S_new U {y} is valid.
                            S = S_new + [y]
                            S_set.remove(x)
                            S_set.add(y)
                            improved = True
                            break
                    if improved:
                        break
        return S

    best_size = 0
    best_set = []
    
    # Define different ordering keys
    def key_desc(x): return -x
    def key_asc(x): return x
    def key_prime_factor_count(x):
        # Count prime factors with multiplicity
        count = 0
        n = x
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    def key_largest_prime_factor(x):
        # Largest prime factor
        n = x
        d = 2
        max_p = 1
        while d * d <= n:
            while n % d == 0:
                max_p = d
                n //= d
            d += 1
        if n > 1:
            max_p = n
        return max_p

    keys = [key_desc, key_asc, key_prime_factor_count, key_largest_prime_factor]
    
    for i, key in enumerate(keys):
        if time.time() - t0 > DEADLINE - 10.0:
            break
        S = greedy_construct(key, seed=i)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Run local search on the best set found
    if best_size > 0 and time.time() - t0 < DEADLINE - 5.0:
        best_set = local_search(best_set)
        best_size = len(best_set)
        
    # Try a few random restarts with local search if time permits
    random.seed(42)
    for i in range(20):
        if time.time() - t0 > DEADLINE - 5.0:
            break
        # Random ordering
        S = greedy_construct(lambda x: random.random(), seed=100+i)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
        # Local search on this random result
        if time.time() - t0 < DEADLINE - 5.0:
            S_ls = local_search(S)
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best_set = S_ls

    result = sorted(best_set)
    return [int(x) for x in result]
