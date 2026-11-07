import time
import random
from itertools import combinations
from math import gcd

DEADLINE = 50.0

def _prime_factorize(n):
    """Return list of prime factors of n."""
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

def _divides_product(a, b, c):
    """Check if a divides b*c using prime factorization."""
    # a divides b*c iff for every prime p, v_p(a) <= v_p(b) + v_p(c)
    # We can check this by computing gcd(a, b) first, then checking if a/gcd(a,b) divides c
    g = gcd(a, b)
    return (a // g) % c == 0 or c % (a // g) == 0

def _valid_add(C, a):
    """Check if adding a to set C maintains the no-three-distinct-members condition."""
    n = len(C)
    for i in range(n):
        b = C[i]
        for j in range(i + 1, n):
            c = C[j]
            # Triple (a, b, c): check none divides product of other two
            if _divides_product(a, b, c):
                return False
            if _divides_product(b, a, c):
                return False
            if _divides_product(c, a, b):
                return False
    return True

def _is_valid_set(S):
    """Check if set S satisfies the condition for all triples."""
    S_list = sorted(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            for k in range(j + 1, n):
                c = S_list[k]
                if _divides_product(a, b, c):
                    return False
                if _divides_product(b, a, c):
                    return False
                if _divides_product(c, a, b):
                    return False
    return True

def construct():
    t0 = time.time()
    universe = list(range(2, 257))
    
    def is_prime(x):
        if x < 2: return False
        if x < 4: return True
        if x % 2 == 0: return False
        i = 3
        while i * i <= x:
            if x % i == 0: return False
            i += 2
        return True
        
    primes = [x for x in universe if is_prime(x)]
    
    best_size = 0
    best_set = set()
    
    def solve_for_P(P, order='desc'):
        """Solve with fixed prime set P included, fill rest greedily."""
        prod_P = 1
        for p in P:
            prod_P *= p
            
        # Numbers coprime to product of P, excluding P itself
        U = [x for x in universe if x not in P and gcd(x, prod_P) == 1]
        
        if order == 'desc':
            U_sorted = sorted(U, reverse=True)
        elif order == 'asc':
            U_sorted = sorted(U)
        elif order == 'random':
            U_sorted = U[:]
            random.shuffle(U_sorted)
        else:
            U_sorted = sorted(U, reverse=True)
            
        C = []
        for x in U_sorted:
            if time.time() - t0 > DEADLINE:
                break
            if _valid_add(C, x):
                C.append(x)
                
        S = set(P) | set(C)
        
        # Local search: try single-element swaps
        if time.time() - t0 < DEADLINE - 5:
            improved = True
            while improved and time.time() - t0 < DEADLINE - 2:
                improved = False
                S_list = sorted(S)
                n = len(S_list)
                for i in range(n):
                    if time.time() - t0 > DEADLINE - 2:
                        break
                    a = S_list[i]
                    # Try removing a and adding some other number
                    S_without_a = S - {a}
                    for x in universe:
                        if x in S_without_a:
                            continue
                        if time.time() - t0 > DEADLINE - 2:
                            break
                        if _valid_add(list(S_without_a), x):
                            new_S = S_without_a | {x}
                            if len(new_S) > len(S):
                                S = new_S
                                improved = True
                                break
                    if improved:
                        break
                        
        return S
    
    # Try specific P sets with different orderings
    P_candidates = [
        set(),
        {2},
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 5},
        {3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
    ]
    
    for P in P_candidates:
        for order in ['desc', 'asc']:
            if time.time() - t0 > DEADLINE:
                break
            S = solve_for_P(P, order)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    # Try random P subsets with random ordering
    random.seed(42)
    for _ in range(30):
        if time.time() - t0 > DEADLINE:
            break
        P = set(random.sample(primes, random.randint(0, 6)))
        S = solve_for_P(P, 'random')
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Final local search on best set
    if time.time() - t0 < DEADLINE - 3:
        S_list = sorted(best_set)
        improved = True
        while improved and time.time() - t0 < DEADLINE - 1:
            improved = False
            for i in range(len(S_list)):
                if time.time() - t0 > DEADLINE - 1:
                    break
                a = S_list[i]
                S_without_a = best_set - {a}
                for x in universe:
                    if x in S_without_a:
                        continue
                    if time.time() - t0 > DEADLINE - 1:
                        break
                    if _valid_add(list(S_without_a), x):
                        new_S = S_without_a | {x}
                        if len(new_S) > len(best_set):
                            best_set = new_S
                            improved = True
                            break
                if improved:
                    break
                    
    result = sorted(best_set)
    return [int(x) for x in result]
