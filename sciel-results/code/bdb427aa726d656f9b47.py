import time
import random
from math import gcd
from itertools import combinations

def construct():
    N = 128
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    # Precompute GCD table
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            G[i][j] = gcd(i, j)
            G[j][i] = G[i][j]
    
    # Helper to check if a set is valid
    def is_valid(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                gab = G[a][b]
                for k in range(j + 1, len(S_list)):
                    c = S_list[k]
                    if G[b][c] == gab and G[c][a] == gab:
                        return False
        return True

    # Helper to check if x can be added to S
    def can_add(S, x):
        # Check all pairs in S
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                gab = G[a][b]
                if G[b][x] == gab and G[x][a] == gab:
                    return False
        return True

    # Greedy build with a specific order
    def greedy_build(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    # Local search
    def local_search(S, time_limit=2.0):
        S = set(S)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # 1. Try to add
            for x in range(1, N + 1):
                if x not in S:
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        break
            # 2. If no add, try swap
            if not improved:
                # Try swapping one out, one in
                # To make this faster, we can limit swaps or randomize
                for u in list(S):
                    S_temp = S - {u}
                    for v in range(1, N + 1):
                        if v not in S:
                            if can_add(S_temp, v):
                                S = S_temp | {v}
                                improved = True
                                break
                    if improved:
                        break
        return S

    best_set = set()
    
    # Heuristic 1: Sort by number of divisors (ascending)
    def count_divisors(n):
        c = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                c += 1
                if i * i != n:
                    c += 1
            i += 1
        return c

    div_counts = [count_divisors(i) for i in range(N + 1)]
    
    # Order 1: Fewest divisors first
    order1 = sorted(range(1, N + 1), key=lambda x: (div_counts[x], x))
    
    # Order 2: Most divisors last (same as above essentially for greedy)
    # Let's try a random restart approach with the best heuristic
    
    # Initial greedy
    S = greedy_build(order1)
    S = local_search(S, time_limit=5.0)
    if len(S) > len(best_set):
        best_set = set(S)
        
    # Randomized restarts
    # We can perturb the order slightly
    base_order = list(range(1, N + 1))
    
    restarts = 0
    while time.time() < deadline:
        # Create a perturbed order
        # Mix of heuristic and random
        if restarts % 2 == 0:
            # Heuristic order with slight noise
            order = sorted(range(1, N + 1), key=lambda x: (div_counts[x] + random.random(), x))
        else:
            # Pure random
            order = base_order[:]
            random.shuffle(order)
            
        S = greedy_build(order)
        S = local_search(S, time_limit=1.5)
        
        if len(S) > len(best_set):
            best_set = set(S)
            
        restarts += 1
        
        # If we found a very large set, we can be more conservative with time per restart
        if len(best_set) > 85:
             # Tighten local search time to allow more restarts? 
             # Or just keep going.
             pass

    return list(best_set) if best_set else [1]
