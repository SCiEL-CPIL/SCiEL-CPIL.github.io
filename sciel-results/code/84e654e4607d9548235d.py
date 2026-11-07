import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    
    n = 256
    candidates = list(range(2, n + 1))
    
    # Sieve
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n+1) if is_prime[i]]
    composites = [i for i in range(2, n+1) if not is_prime[i]]
    
    # Precompute: for each a, set of (b,c) with b!=c such that a | b*c
    # Use bitsets or just sets for efficiency
    # conflict[a] = set of frozenset({b,c}) or just a set of pairs
    # For can_add(S, x): need to check for all a in S, (x,b) not in conflict[a] for all b in S, b!=a
    # and for all b in S, (a,x) not in conflict[b]... wait this is getting complicated.
    
    # Simpler: precompute for each pair (u,v) with u!=v, the set of a's that divide u*v
    # divisors_of_product[u][v] = set of a in [2,256] such that a | u*v
    # Then for set S, it's valid iff for all distinct a,b,c in S: a not in divisors_of_product[b][c]
    
    # This is 255*254/2 = 32385 pairs, each with up to 255 divisors. Memory: ~8M entries. OK.
    
    # Actually, let's just precompute a 3D boolean array or use a dict
    # divs[a][b][c] would be 256^3 = 16M booleans. Use numpy.
    
    import numpy as np
    
    # divs[a-2, b-2, c-2] = 1 if a | b*c, for a,b,c in [2,256]
    # Size: 255^3 = 16.5M bytes as uint8. OK.
    
    # But we only need it for distinct a,b,c. Let's compute it.
    # Actually, for can_add and is_valid, we'll just do the modulo checks directly.
    # With m <= 60, m^3 = 216000, which is ~0.01s in Python. Fine for a few hundred calls.
    
    def is_valid(S_list):
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(m):
                if j == i: continue
                b = S_list[j]
                for k in range(m):
                    if k == i or k == j: continue
                    if (b * S_list[k]) % a == 0:
                        return False
        return True
    
    def can_add(S_set, x):
        if x in S_set: return False
        S_list = list(S_set)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(m):
                if j == i: continue
                b = S_list[j]
                # x, a, b are distinct (x not in S_set)
                if (a * b) % x == 0: return False
                if (x * b) % a == 0: return False
                if (x * a) % b == 0: return False
        return True
    
    # Greedy
    def greedy(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S
    
    best_S = set(primes)
    
    for order in [primes + composites, composites + primes,
                  sorted(candidates), sorted(candidates, reverse=True),
                  sorted(candidates, key=lambda x: x**0.5)]:
        S = greedy(order)
        if len(S) > len(best_S):
            best_S = S
    
    # Local search
    random.seed(42)
    stall = 0
    while time.time() < deadline and stall < 100:
        improved = False
        S_list = list(best_S)
        
        # Additions
        for x in candidates:
            if x not in best_S and can_add(best_S, x):
                best_S.add(x)
                improved = True
                break
        
        if time.time() > deadline: break
        
        # Swaps: try replacing one element with another
        for i in range(len(S_list)):
            if time.time() > deadline: break
            rem = S_list[i]
            for add in candidates:
                if add in best_S: continue
                new_S = best_S - {rem} | {add}
                if is_valid(list(new_S)):
                    # Check if this enables more additions
                    for z in candidates:
                        if z not in new_S and can_add(new_S, z):
                            best_S = new_S
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        
        if not improved:
            stall += 1
    
    result = sorted(best_S)
    if not is_valid(result):
        result = primes
    return [int(x) for x in result]
