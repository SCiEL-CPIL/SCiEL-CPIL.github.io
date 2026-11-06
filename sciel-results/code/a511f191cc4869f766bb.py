import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute bad pairs for each element
    # For each x, store the set of pairs (u,v) with u<v such that {x,u,v} is bad
    # Also precompute a conflict matrix: conflict[x][y] = set of z such that {x,y,z} is bad
    bad_pairs_for = [set() for _ in range(n + 1)]
    # conflict[x][y] = set of z > max(x,y) such that {x,y,z} is bad
    conflict = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    
    # Precompute all bad triples
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    # Triple (a, b, c) is bad
                    bad_pairs_for[a].add((b, c))
                    bad_pairs_for[b].add((a, c))
                    bad_pairs_for[c].add((a, b))
                    # For conflict matrix: for pair (a,b), c conflicts
                    conflict[a][b].add(c)
                    conflict[b][a].add(c)
    
    def can_add_fast(x, current_set):
        """Check if x can be added to current_set using conflict matrix."""
        # x can be added if for every pair (u,v) in current_set, 
        # it's not the case that {x,u,v} is a bad triple
        # Equivalently: for every u in current_set, no v in current_set with v in conflict[x][u]
        # But this is O(|S|^2). Better: for each u in current_set, check if conflict[x][u] intersects current_set
        # Even better: precompute for each x, the set of all elements that form a bad triple with x and some other element
        # Actually, let's use the bad_pairs_for approach but with a faster check
        
        # For each pair (u,v) in bad_pairs_for[x], check if both u and v are in current_set
        for (u, v) in bad_pairs_for[x]:
            if u in current_set and v in current_set:
                return False
        return True

    def greedy(order):
        selected = set()
        for x in order:
            if can_add_fast(x, selected):
                selected.add(x)
        return selected

    def local_search(S):
        improved = True
        S = set(S)
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add any element - most important
            for x in range(1, n + 1):
                if x not in S and can_add_fast(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if not improved:
                # 2. Try 1-for-2 swap: remove x, add two new elements
                found_swap = False
                s_list = list(S)
                random.shuffle(s_list)  # Randomize order for diversity
                for x in s_list:
                    if time.time() > deadline - 2:
                        break
                    S_trial = S - {x}
                    # Try to add two elements greedily
                    added = []
                    for y in range(1, n + 1):
                        if y not in S_trial and can_add_fast(y, S_trial):
                            S_trial.add(y)
                            added.append(y)
                            if len(added) == 2:
                                break
                    if len(added) == 2:
                        S = S_trial
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    continue

            if not improved:
                # 3. Try 2-for-3 swap (more expensive, but can escape deeper local minima)
                found_swap = False
                s_list = list(S)
                random.shuffle(s_list)
                for i in range(len(s_list)):
                    if time.time() > deadline - 2:
                        break
                    for j in range(i + 1, len(s_list)):
                        S_trial = S - {s_list[i], s_list[j]}
                        added = []
                        for y in range(1, n + 1):
                            if y not in S_trial and can_add_fast(y, S_trial):
                                S_trial.add(y)
                                added.append(y)
                                if len(added) == 3:
                                    break
                        if len(added) == 3:
                            S = S_trial
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    continue
        
        return S

    best_set = set()
    best_size = 0
    
    # Precompute degrees for ordering
    degrees = [len(bad_pairs_for[k]) for k in range(n+1)]
    
    # Run multiple randomized greedy passes
    num_restarts = 150
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different ordering strategies
        if i % 5 == 0:
            # Pure random
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif i % 5 == 1:
            # Sort by degree (least constrained first) with noise
            order = list(range(1, n + 1))
            order.sort(key=lambda x: degrees[x] + random.random() * 0.5)
        elif i % 5 == 2:
            # Sort by degree descending (most constrained first)
            order = list(range(1, n + 1))
            order.sort(key=lambda x: -degrees[x] + random.random() * 0.5)
        elif i % 5 == 3:
            # Random with bias toward low-degree elements
            order = list(range(1, n + 1))
            # Weighted shuffle: lower degree = higher chance to come first
            weights = [1.0 / (degrees[x] + 1) for x in order]
            for pos in range(len(order) - 1, 0, -1):
                total = sum(weights[:pos + 1])
                r = random.random() * total
                cum = 0
                for k in range(pos + 1):
                    cum += weights[k]
                    if r <= cum:
                        order[pos], order[k] = order[k], order[pos]
                        weights[pos], weights[k] = weights[k], weights[pos]
                        break
        else:
            # Reverse sorted by value
            order = list(range(n, 0, -1))
            random.shuffle(order)
        
        S = greedy(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Final polish on best_set
    best_set = local_search(best_set)
    
    return list(best_set)
