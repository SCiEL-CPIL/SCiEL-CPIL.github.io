import time
from math import gcd, isqrt
from itertools import combinations

N = 512
DEADLINE = 85.0  # Safety margin before 90s kill

def is_compatible(x, S):
    for y in S:
        if (x // gcd(x, y)) * y > N:
            return False
    return True

def get_divisors(n):
    divs = []
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

def construct():
    start_time = time.time()
    
    # Find M <= 512 with max divisors
    best_M = 1
    best_tau = 0
    for M in range(1, N + 1):
        tau = len(get_divisors(M))
        if tau > best_tau:
            best_tau = tau
            best_M = M
            
    # Start with divisors of best_M
    best_S = set(get_divisors(best_M))
    best_size = len(best_S)
    
    # Greedy add: try to add any compatible x
    for x in range(1, N + 1):
        if x not in best_S and is_compatible(x, best_S):
            best_S.add(x)
            best_size += 1
    
    # Local Search: Try to increase size by swapping
    # Strategy: Remove k elements, add k+1 elements
    # We focus on k=1 (remove 1, add 2) and k=2 (remove 2, add 3)
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        
        # Try remove 1, add 2
        for y in list(best_S):
            if time.time() > DEADLINE:
                break
            S_temp = best_S - {y}
            
            # Find candidates compatible with S_temp
            candidates = []
            for x in range(1, N + 1):
                if x not in S_temp and is_compatible(x, S_temp):
                    candidates.append(x)
            
            # Find pair x1, x2 in candidates such that they are compatible
            # and both can be added to S_temp
            found_pair = False
            for i in range(len(candidates)):
                if time.time() > DEADLINE:
                    break
                x1 = candidates[i]
                S_with_x1 = S_temp | {x1}
                for j in range(i + 1, len(candidates)):
                    x2 = candidates[j]
                    if is_compatible(x2, S_with_x1):
                        new_S = S_with_x1 | {x2}
                        if len(new_S) > best_size:
                            best_S = new_S
                            best_size = len(best_S)
                            improved = True
                            found_pair = True
                            break
                if found_pair:
                    break
            if found_pair:
                break  # Restart loop with new best_S

        if improved:
            continue

        # Try remove 2, add 3
        # This is more expensive. Only if time permits.
        if time.time() < DEADLINE - 10:
            found_triple = False
            for pair in combinations(list(best_S), 2):
                if time.time() > DEADLINE:
                    break
                y1, y2 = pair
                S_temp = best_S - {y1, y2}
                
                # Find candidates compatible with S_temp
                candidates = []
                for x in range(1, N + 1):
                    if x not in S_temp and is_compatible(x, S_temp):
                        candidates.append(x)
                
                # Try to add 3 greedily from candidates
                S_new = S_temp.copy()
                added = 0
                for x in candidates:
                    if is_compatible(x, S_new):
                        S_new.add(x)
                        added += 1
                        if added == 3:
                            break
                
                if len(S_new) > best_size:
                    best_S = S_new
                    best_size = len(best_S)
                    improved = True
                    found_triple = True
                    break
            
            if found_triple:
                continue

        # If no improvement from swaps, try random restarts with different seeds
        # to explore different local optima
        if time.time() < DEADLINE - 20:
            import random
            random.seed(int(time.time() * 1000) % 10000)
            S_rand = set()
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates:
                if is_compatible(x, S_rand):
                    S_rand.add(x)
            if len(S_rand) > best_size:
                best_S = S_rand
                best_size = len(best_S)
                improved = True

    return list(best_S)
