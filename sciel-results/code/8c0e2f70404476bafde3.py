import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    LIMIT = 512
    
    # Precompute compatibility matrix
    # compat[i][j] = True if lcm(i+1, j+1) <= LIMIT
    # Use 1-indexed: index 0 corresponds to value 1, etc.
    n = LIMIT
    # We'll use a bitset or just check on the fly with precomputed gcd
    
    # Precompute lcm for all pairs is O(n^2) = 512^2 = 262144, which is fast
    # But let's precompute a compatibility matrix as a list of sets for speed
    # For each x, which y are compatible with x?
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Precompute compatible partners for each number 1..512
    # compat[x] = set of y such that lcm(x,y) <= 512
    compat = [set() for _ in range(LIMIT + 1)]
    for x in range(1, LIMIT + 1):
        for y in range(x, LIMIT + 1):
            l = x * y // gcd(x, y)
            if l <= LIMIT:
                compat[x].add(y)
                if y != x:
                    compat[y].add(x)
    
    # Now we have compat[x] as the set of all y compatible with x
    
    # Strategy: Greedy + Local Search with random restarts
    
    def greedy_build(seed_order):
        """Build a set by greedily adding elements in the given order."""
        S = set()
        for x in seed_order:
            if all(lcm(x, y) <= LIMIT for y in S):
                S.add(x)
        return S
    
    def local_search(S, max_iterations=10000):
        """Improve set S via local search: try swaps and additions."""
        S = set(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            # Try to add any element not in S
            for x in range(1, LIMIT + 1):
                if x in S:
                    continue
                if all(lcm(x, y) <= LIMIT for y in S):
                    S.add(x)
                    improved = True
                    break
            if improved:
                continue
            # Try swaps: remove one element, add another
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() > deadline - 5:
                    return S
                x_remove = S_list[i]
                S_temp = S - {x_remove}
                for x_add in range(1, LIMIT + 1):
                    if x_add in S_temp:
                        continue
                    if all(lcm(x_add, y) <= LIMIT for y in S_temp):
                        S.add(x_add)
                        improved = True
                        break
                if improved:
                    break
        return S
    
    # Initial greedy: sort by number of compatible partners (fewest first, more constrained first)
    # Or sort by value
    # Let's try multiple strategies
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy with natural order
    S1 = greedy_build(range(1, LIMIT + 1))
    S1 = local_search(S1, max_iterations=5000)
    if len(S1) > best_size:
        best_set = S1
        best_size = len(S1)
    
    # Strategy 2: Greedy with reverse order
    S2 = greedy_build(range(LIMIT, 0, -1))
    S2 = local_search(S2, max_iterations=5000)
    if len(S2) > best_size:
        best_set = S2
        best_size = len(S2)
    
    # Strategy 3: Greedy sorted by number of compatible partners (ascending - most constrained first)
    num_compat = [len(compat[x]) for x in range(1, LIMIT + 1)]
    order3 = sorted(range(1, LIMIT + 1), key=lambda x: num_compat[x])
    S3 = greedy_build(order3)
    S3 = local_search(S3, max_iterations=5000)
    if len(S3) > best_size:
        best_set = S3
        best_size = len(S3)
    
    # Strategy 4: Greedy sorted by number of compatible partners (descending)
    order4 = sorted(range(1, LIMIT + 1), key=lambda x: -num_compat[x])
    S4 = greedy_build(order4)
    S4 = local_search(S4, max_iterations=5000)
    if len(S4) > best_size:
        best_set = S4
        best_size = len(S4)
    
    # Random restarts
    import random
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        # Random permutation
        order = list(range(1, LIMIT + 1))
        random.shuffle(order)
        S = greedy_build(order)
        S = local_search(S, max_iterations=3000)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Final local search on best set
    if time.time() < deadline - 5:
        best_set = local_search(best_set, max_iterations=10000)
    
    return sorted(best_set)
