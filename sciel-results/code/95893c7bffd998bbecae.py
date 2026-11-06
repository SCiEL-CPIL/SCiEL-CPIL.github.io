"""Improved program: maximal independent set in lcm-incompatibility graph."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 45  # safety margin before 60s limit
    limit = 512
    n = limit  # elements are 1..512, index 0 unused
    
    # Precompute lcm compatibility matrix
    # compat[i][j] = True iff lcm(i, j) <= 512
    # We'll use a bitset approach for speed
    
    # First, compute which numbers are compatible with each other
    # lcm(a,b) = a*b // gcd(a,b) <= 512  <=>  a*b <= 512 * gcd(a,b)
    
    # Build adjacency: for each pair (i,j), i<j, check if lcm > 512
    # This is an edge in the "conflict graph". We want a maximum independent set.
    
    # Precompute gcd table or just compute on the fly
    # 512*512/2 = ~131072 pairs, very fast
    
    # Let's represent the conflict graph as a list of sets
    # conflicts[i] = set of j such that lcm(i,j) > 512
    conflicts = [set() for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            if i * j // g > limit:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Greedy + local search for maximum independent set
    # Start with a greedy independent set
    # Then try to improve via swaps and additions
    
    # Greedy: process vertices in some order, add if compatible with all current members
    # Order: try vertices with fewer conflicts first (more likely to fit)
    # Actually, let's try multiple orders and take the best
    
    def greedy_order(order):
        """Build independent set by adding vertices in given order if compatible."""
        S = set()
        for v in order:
            # Check if v is compatible with all in S
            compatible = True
            for u in S:
                if v in conflicts[u]:
                    compatible = False
                    break
            if compatible:
                S.add(v)
        return S
    
    # Try several heuristics for ordering
    best_set = set()
    
    # Heuristic 1: vertices with smallest degree (fewest conflicts) first
    degrees = [len(conflicts[i]) for i in range(n + 1)]
    order1 = sorted(range(1, n + 1), key=lambda x: degrees[x])
    S1 = greedy_order(order1)
    if len(S1) > len(best_set):
        best_set = S1
    
    # Heuristic 2: vertices with largest value first (might allow more room)
    order2 = list(range(n, 0, -1))
    S2 = greedy_order(order2)
    if len(S2) > len(best_set):
        best_set = S2
    
    # Heuristic 3: vertices with smallest value first
    order3 = list(range(1, n + 1))
    S3 = greedy_order(order3)
    if len(S3) > len(best_set):
        best_set = S3
    
    # Heuristic 4: random order (a few tries)
    import random
    rng = random.Random(42)
    for _ in range(50):
        if time.time() > deadline - 15:
            break
        order4 = list(range(1, n + 1))
        rng.shuffle(order4)
        S4 = greedy_order(order4)
        if len(S4) > len(best_set):
            best_set = S4
    
    # Local search: try to add vertices that are currently excluded
    # For each vertex not in best_set, try to add it (remove conflicting members if beneficial)
    # Simpler: try to add any vertex that is compatible with all in best_set
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for v in range(1, n + 1):
            if v not in best_set:
                # Check if v can be added
                can_add = True
                for u in best_set:
                    if v in conflicts[u]:
                        can_add = False
                        break
                if can_add:
                    best_set.add(v)
                    improved = True
    
    # Try swap-based local search: remove one element, try to add one or more
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for v in list(best_set):
                # Try removing v
                candidate = best_set - {v}
                # Try to add any vertex not in candidate that is compatible
                added = False
                for w in range(1, n + 1):
                    if w not in candidate and w != v:
                        can_add = True
                        for u in candidate:
                            if w in conflicts[u]:
                                can_add = False
                                break
                        if can_add:
                            candidate.add(w)
                            added = True
                            break
                if added and len(candidate) > len(best_set):
                    best_set = candidate
                    improved = True
    
    # Final check: verify the set is valid
    result = sorted(best_set)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if a * b // gcd(a, b) > limit:
                # Should not happen, but fallback to seed
                return list(range(1, 23))
    
    if not result:
        return [1]
    
    return result
# EVOLVE-BLOCK-END
