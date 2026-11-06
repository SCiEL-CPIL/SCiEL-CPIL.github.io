import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    def is_bad_triple(a, b, c):
        """Check if three distinct elements have all three pairwise GCDs equal."""
        ga = gcd_table[a][b]
        gb = gcd_table[a][c]
        gc = gcd_table[b][c]
        return (ga == gb == gc)
    
    def find_bad_triples(S):
        """Find all bad triples in set S. Returns list of tuples."""
        triples = []
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                ga = gcd_table[a][b]
                for k in range(j+1, n):
                    c = S_list[k]
                    if gcd_table[a][c] == ga and gcd_table[b][c] == ga:
                        triples.append((a, b, c))
        return triples
    
    def can_add(x, S):
        """Check if x can be added to S without creating a bad triple."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            ga = gcd_table[a][x]
            for j in range(i+1, n):
                b = S_list[j]
                if gcd_table[a][b] == ga and gcd_table[b][x] == ga:
                    return False
        return True

    def prune_and_optimize(initial_set, rng):
        """
        Start with initial_set, prune bad triples by removing high-degree nodes,
        then try to add back elements.
        """
        S = set(initial_set)
        
        # Phase 1: Prune
        while True:
            if time.time() > deadline - 1.0:
                break
            triples = find_bad_triples(S)
            if not triples:
                break
            
            # Count frequency of each element in bad triples
            freq = {}
            for a, b, c in triples:
                freq[a] = freq.get(a, 0) + 1
                freq[b] = freq.get(b, 0) + 1
                freq[c] = freq.get(c, 0) + 1
            
            # Find max frequency
            max_freq = max(freq.values())
            candidates = [x for x, f in freq.items() if f == max_freq]
            
            # Randomly pick one to break symmetry and explore different paths
            x_to_remove = rng.choice(candidates)
            S.remove(x_to_remove)
        
        # Phase 2: Add back elements
        changed = True
        while changed and time.time() < deadline - 0.5:
            changed = False
            for x in range(1, N+1):
                if x in S:
                    continue
                if can_add(x, S):
                    S.add(x)
                    changed = True
        
        # Phase 3: Local Search (Swaps)
        improved = True
        while improved and time.time() < deadline - 0.2:
            improved = False
            S_list = list(S)
            for x in S_list:
                if time.time() > deadline - 0.2:
                    break
                S.remove(x)
                for y in range(1, N+1):
                    if y in S or y == x:
                        continue
                    if can_add(y, S):
                        S.add(y)
                        improved = True
                        break
                if improved:
                    break
                S.add(x) # restore x if no swap found

        return S

    best_set = set()
    
    # Run multiple trials with different random seeds
    # Estimate time per trial. If it's too slow, reduce count.
    for trial in range(100):
        if time.time() > deadline - 2.0:
            break
        
        rng = random.Random(trial)
        
        # Strategy 1: Start with all, prune
        S1 = prune_and_optimize(set(range(1, N+1)), rng)
        if len(S1) > len(best_set):
            best_set = S1
            
        # Strategy 2: Start with random subset to escape local optima
        # Use a smaller initial set to speed up pruning?
        # Or just random shuffle of removal order is already in Strategy 1.
        # Let's try starting with evens?
        if trial % 2 == 1:
            S2 = prune_and_optimize(set(range(2, N+1, 2)), rng)
            if len(S2) > len(best_set):
                best_set = S2
        else:
            S3 = prune_and_optimize(set(range(1, N+1, 2)), rng)
            if len(S3) > len(best_set):
                best_set = S3

    return list(best_set) if best_set else [1]
