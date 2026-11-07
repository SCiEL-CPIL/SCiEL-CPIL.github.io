import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    N = 256
    candidates = list(range(2, N + 1))
    num_candidates = N - 1  # 255 integers
    
    # 1. Identify primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N+1) if is_prime[i]]
    
    # Strategy: Large Primes + Small Smooth Composites
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
            for p in small_primes:
                while m % p == 0:
                    m //= p
            if m == 1:
                smooth_candidates.append(n)
    
    # Precompute conflict bitsets for smooth candidates
    # conflict_mask[i] is a bitset where bit j is set if candidate i conflicts with candidate j
    # Conflict: for distinct a, b, c in set, a does not divide b*c
    # When adding y to S, y conflicts with pair (u,v) in S if:
    #   - y divides u*v (y is the divisor)
    #   - u divides y*v (u is the divisor, y is one of the pair)
    #   - v divides y*u (v is the divisor, y is one of the pair)
    
    num_smooth = len(smooth_candidates)
    smooth_set = set(smooth_candidates)
    
    # For each smooth candidate, compute which other smooth candidates it conflicts with
    # in the sense that they cannot be in the same valid set
    # Two elements a, b conflict if there exists c such that a|b*c or b|a*c
    # But we need triple conflicts, not just pairs.
    # Actually, the constraint is on triples. Two elements can coexist if no third element causes a violation.
    # This is a 3-uniform hypergraph problem.
    
    # Let's use a different approach: greedy + local search with bitset acceleration
    
    def is_valid_subset(subset):
        s = set(subset)
        items = list(s)
        n = len(items)
        for i in range(n):
            a = items[i]
            for j in range(n):
                if i == j: continue
                b = items[j]
                for k in range(n):
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

    # Initial Greedy: sort by number of conflicts (degree in conflict graph)
    # Compute conflict degree for each smooth candidate
    # Two candidates conflict if they cannot be in a valid set together with some third element
    # For simplicity, use a proxy: count how many other candidates x have y|x*z or x|y*z for some z
    
    # Actually, let's use a simpler greedy: sort by value descending (larger numbers are "safer")
    best_small_set = []
    ordered = sorted(smooth_candidates, reverse=True)
    for n in ordered:
        if time.time() > deadline:
            break
        test_set = best_small_set + [n]
        if is_valid_subset(test_set):
            best_small_set = test_set
    
    # Local Search: 1-for-1 swaps and 2-for-1 swaps
    S = set(best_small_set)
    C = [x for x in smooth_candidates if x not in S]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S_list = list(S)
        
        # Try 1-for-1 swaps
        for x in S_list:
            if time.time() > deadline - 5:
                break
            S_minus_x = S - {x}
            for y in C:
                if is_valid_addition(S_minus_x, y):
                    # Swap x for y
                    S = S_minus_x | {y}
                    C = [c for c in C if c != y] + [x]
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-for-1 swaps
        found_swap = False
        for x in S_list:
            if time.time() > deadline - 5:
                break
            S_minus_x = S - {x}
            
            valid_ys = []
            for y in C:
                if is_valid_addition(S_minus_x, y):
                    valid_ys.append(y)
            
            for y in valid_ys:
                S_prime = S_minus_x | {y}
                for z in C:
                    if z == y:
                        continue
                    if is_valid_addition(S_prime, z):
                        S = S_prime | {z}
                        C = [c for c in C if c not in {y, z}] + [x]
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                break
                
    best_small_set = list(S)
    
    # Combine with large primes
    final_set = list(set(large_primes + best_small_set))
    
    # Verify final set
    if not is_valid_subset(final_set):
        # Fallback: just use large primes
        final_set = large_primes
    
    # Ensure types are correct
    return [int(x) for x in final_set]
