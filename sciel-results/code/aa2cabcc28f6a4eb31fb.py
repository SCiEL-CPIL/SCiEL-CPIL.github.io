import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128
    nums = list(range(1, n + 1))
    
    # 1. Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # 2. Identify all bad triples
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = set()
    for a, b, c in combinations(nums, 3):
        if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
            bad_triples.add((a, b, c))
    
    # 3. Build conflict map: for each pair (a,b), store the set of c's that form a bad triple with them
    # This allows O(1) check if adding x creates a bad triple with existing pair (a,b)
    # conflict_map[a][b] = set of c such that (a,b,c) is a bad triple
    conflict_map = {}
    for a, b, c in bad_triples:
        if (a, b) not in conflict_map:
            conflict_map[(a, b)] = set()
        conflict_map[(a, b)].add(c)
        
        if (b, a) not in conflict_map:
            conflict_map[(b, a)] = set()
        conflict_map[(b, a)].add(c)
        
        if (a, c) not in conflict_map:
            conflict_map[(a, c)] = set()
        conflict_map[(a, c)].add(b)
        
        if (c, a) not in conflict_map:
            conflict_map[(c, a)] = set()
        conflict_map[(c, a)].add(b)
        
        if (b, c) not in conflict_map:
            conflict_map[(b, c)] = set()
        conflict_map[(b, c)].add(a)
        
        if (c, b) not in conflict_map:
            conflict_map[(c, b)] = set()
        conflict_map[(c, b)].add(a)

    # 4. Calculate degree of each vertex (number of bad triples it is part of)
    degree = {x: 0 for x in nums}
    for a, b, c in bad_triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1

    def can_add(x, S_set):
        """Check if x can be added to S without creating a bad triple."""
        s_list = list(S_set)
        m = len(s_list)
        for i in range(m):
            a = s_list[i]
            for j in range(i + 1, m):
                b = s_list[j]
                # Check if (a, b, x) is a bad triple
                # We can check if x is in conflict_map[(a,b)]
                if x in conflict_map.get((a, b), set()):
                    return False
        return True

    def greedy_build(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S, remaining):
        """Try to improve S by adding elements or swapping."""
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding any remaining element
            for x in remaining:
                if time.time() > deadline - 2:
                    break
                if can_add(x, S):
                    S.add(x)
                    remaining.discard(x)
                    improved = True
            # If no additions, try swaps
            if not improved:
                s_list = list(S)
                for a in s_list:
                    if time.time() > deadline - 2:
                        break
                    S_minus_a = S - {a}
                    for x in remaining:
                        if can_add(x, S_minus_a):
                            S.discard(a)
                            S.add(x)
                            remaining.discard(x)
                            remaining.add(a)
                            improved = True
                            break
                    if improved:
                        break
        return S

    best_size = 0
    best_set = set()
    
    # Strategy 1: Greedy by degree (lowest degree first)
    # This tends to pick numbers that are less "central" in the conflict graph
    order_by_degree = sorted(nums, key=lambda x: degree[x])
    S1 = greedy_build(order_by_degree)
    best_size = len(S1)
    best_set = set(S1)
    
    # Strategy 2: Random restarts with greedy
    # We run multiple random restarts to find a better local optimum
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        shuffled = nums[:]
        random.shuffle(shuffled)
        S = greedy_build(shuffled)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Refine the best set found so far
    if time.time() < deadline - 5:
        remaining = set(nums) - best_set
        best_set = local_search(best_set, remaining)
        best_size = len(best_set)
        
    # If we have time, try local search on other promising candidates
    # Specifically, the degree-based one might be good to refine if it wasn't the best
    if time.time() < deadline - 2:
        S_deg = greedy_build(order_by_degree)
        if len(S_deg) > best_size - 2: # If close to best
             remaining_deg = set(nums) - S_deg
             S_deg = local_search(S_deg, remaining_deg)
             if len(S_deg) > best_size:
                 best_size = len(S_deg)
                 best_set = S_deg

    return list(best_set)
