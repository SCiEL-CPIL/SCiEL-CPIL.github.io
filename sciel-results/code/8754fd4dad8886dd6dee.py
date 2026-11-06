import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute bad pairs: for each x, list of (a,b) pairs such that {x,a,b} is a bad triple
    bad_pairs_for = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) != ab:
                    continue
                if _lcm(b, c) == ab:
                    bad_pairs_for[a].add((b, c))
                    bad_pairs_for[b].add((a, c))
                    bad_pairs_for[c].add((a, b))
    
    # Convert to lists for faster iteration
    bad_pairs_list = [list(s) for s in bad_pairs_for]
    
    def can_add(x, current_set):
        for (a, b) in bad_pairs_list[x]:
            if a in current_set and b in current_set:
                return False
        return True

    def greedy(order):
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected

    def local_search(S):
        S = set(S)
        max_iterations = 50
        for _ in range(max_iterations):
            if time.time() > deadline - 3:
                break
            improved = False
            
            # 1. Try to add any element
            for x in range(1, n + 1):
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try 1-for-1 swaps
            for x in list(S):
                S_trial = S - {x}
                for y in range(1, n + 1):
                    if y not in S_trial and can_add(y, S_trial):
                        S_trial.add(y)
                        S = S_trial
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # 3. Try 1-for-2 swaps
            for x in list(S):
                S_trial = S - {x}
                added_count = 0
                for y in range(1, n + 1):
                    if y not in S_trial and can_add(y, S_trial):
                        S_trial.add(y)
                        added_count += 1
                        if added_count >= 2:
                            break
                if len(S_trial) > len(S):
                    S = S_trial
                    improved = True
                    break
            
            if improved:
                continue
            
            # 4. Try 2-for-3 swaps (sampled to avoid O(n^2) per iteration)
            s_list = list(S)
            random.shuffle(s_list)
            max_pairs_to_try = min(len(s_list) * (len(s_list) - 1) // 2, 200)
            pairs_tried = 0
            for i in range(len(s_list)):
                if pairs_tried >= max_pairs_to_try or time.time() > deadline - 3:
                    break
                for j in range(i + 1, len(s_list)):
                    if pairs_tried >= max_pairs_to_try or time.time() > deadline - 3:
                        break
                    pairs_tried += 1
                    S_trial = S - {s_list[i], s_list[j]}
                    added_count = 0
                    for y in range(1, n + 1):
                        if y not in S_trial and can_add(y, S_trial):
                            S_trial.add(y)
                            added_count += 1
                            if added_count >= 3:
                                break
                    if len(S_trial) > len(S):
                        S = S_trial
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return S

    best_set = set()
    best_size = 0
    
    # Precompute degrees for sorting
    degrees = [len(bad_pairs_list[k]) for k in range(n+1)]
    
    # Run multiple randomized greedy passes
    for i in range(300):
        if time.time() > deadline - 15:
            break
        
        # Different strategies for different iterations
        if i % 5 == 0:
            # Pure random
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif i % 5 == 1:
            # Sort by degree (ascending - add less conflicting first)
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] + random.random() * 0.1)
        elif i % 5 == 2:
            # Sort by degree (descending - add more conflicting first, might force better structure)
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] + random.random() * 0.1)
        elif i % 5 == 3:
            # Random with bias toward low-degree elements
            order = list(range(1, n + 1))
            random.shuffle(order)
            order.sort(key=lambda x: degrees[x] * random.random())
        else:
            # Reverse natural order
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
