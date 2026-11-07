import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # 1. Identify primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N+1) if is_prime[i]]
    
    # Strategy A: Large Primes + Small Smooth Composites
    small_primes = [2, 3, 5, 7, 11, 13]
    large_primes = [p for p in primes if p > 13]
    
    # Generate 13-smooth numbers
    smooth_candidates = []
    for n in range(2, N+1):
        if is_prime[n]:
            if n in small_primes:
                smooth_candidates.append(n)
        else:
            m = n
            smooth = True
            for p in small_primes:
                while m % p == 0:
                    m //= p
            if m == 1:
                smooth_candidates.append(n)
            
    def is_valid_subset(subset):
        s = set(subset)
        items = list(s)
        for i in range(len(items)):
            a = items[i]
            for j in range(len(items)):
                if i == j: continue
                b = items[j]
                for k in range(len(items)):
                    if i == k or j == k: continue
                    c = items[k]
                    if (b * c) % a == 0:
                        return False
        return True

    def is_valid_addition(S, y):
        # S is a valid set. y not in S.
        # Check if S U {y} is valid.
        S_list = list(S)
        n = len(S_list)
        
        # 1. y does not divide u*v for any u,v in S, u!=v
        for i in range(n):
            u = S_list[i]
            for j in range(n):
                if i == j: continue
                v = S_list[j]
                if (u * v) % y == 0:
                    return False
                    
        # 2. u does not divide y*v for any u,v in S, u!=v
        for i in range(n):
            u = S_list[i]
            for j in range(n):
                if i == j: continue
                v = S_list[j]
                if (y * v) % u == 0:
                    return False
                    
        return True

    # Initial Greedy
    best_small_set = []
    ordered = sorted(smooth_candidates, reverse=True)
    for n in ordered:
        test_set = best_small_set + [n]
        if is_valid_subset(test_set):
            best_small_set = test_set
            
    # Local Search: 2-for-1 Swaps
    S = set(best_small_set)
    C = [x for x in smooth_candidates if x not in S]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(S)
        found_swap = False
        
        for x in S_list:
            if time.time() > deadline:
                break
            S_minus_x = S - {x}
            
            # Find y in C such that S_minus_x U {y} is valid
            valid_ys = []
            for y in C:
                if is_valid_addition(S_minus_x, y):
                    valid_ys.append(y)
            
            # Try to find z in C, z!=y, such that (S_minus_x U {y}) U {z} is valid
            for y in valid_ys:
                S_prime = S_minus_x | {y}
                for z in C:
                    if z == y:
                        continue
                    if is_valid_addition(S_prime, z):
                        # Success: Swap x for y, z
                        S = S_prime | {z}
                        C = [c for c in C if c not in {y, z}]
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                break
                
    best_small_set = list(S)
    
    # Combine
    final_set = list(set(large_primes + best_small_set))
    
    # Verify final set
    if not is_valid_subset(final_set):
        final_set = large_primes
    else:
        pass

    return final_set
