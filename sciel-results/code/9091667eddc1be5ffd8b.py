import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]
    
    # Precompute bad pairs: for each x, bitmask of elements that conflict
    # bad_pairs_for[x] = set of (a,b) such that {x,a,b} is a bad triple
    bad_pairs_for = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    bad_pairs_for[a].add((b, c))
                    bad_pairs_for[b].add((a, c))
                    bad_pairs_for[c].add((a, b))
    
    # Convert to lists for faster iteration
    bad_pairs_list = [list(s) for s in bad_pairs_for]
    
    def can_add(x, current_mask):
        """Check if x can be added given current_mask (bitmask of selected elements)."""
        for (a, b) in bad_pairs_list[x]:
            if (current_mask >> a) & 1 and (current_mask >> b) & 1:
                return False
        return True
    
    def greedy(order):
        selected_mask = 0
        selected = set()
        for x in order:
            if can_add(x, selected_mask):
                selected_mask |= (1 << x)
                selected.add(x)
        return selected
    
    def local_search(S, max_iterations=30):
        S = set(S)
        current_size = len(S)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # 1. Try to add any single element
            for x in range(1, n + 1):
                if x not in S and can_add(x, 0):
                    # Check against actual S
                    ok = True
                    for (a, b) in bad_pairs_list[x]:
                        if a in S and b in S:
                            ok = False
                            break
                    if ok:
                        S.add(x)
                        current_size += 1
                        improved = True
                        break
            
            if improved:
                continue
            
            # 2. Try 1-for-1 swaps
            s_list = list(S)
            random.shuffle(s_list)
            for x in s_list:
                S_trial = S - {x}
                for y in range(1, n + 1):
                    if y not in S_trial:
                        ok = True
                        for (a, b) in bad_pairs_list[y]:
                            if a in S_trial and b in S_trial:
                                ok = False
                                break
                        if ok:
                            S = S_trial
                            S.add(y)
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue
            
            # 3. Try 1-for-2 swaps
            for x in s_list:
                S_trial = S - {x}
                added = 0
                for y in range(1, n + 1):
                    if y not in S_trial:
                        ok = True
                        for (a, b) in bad_pairs_list[y]:
                            if a in S_trial and b in S_trial:
                                ok = False
                                break
                        if ok:
                            S_trial.add(y)
                            added += 1
                            if added >= 2:
                                break
                if len(S_trial) > current_size:
                    S = S_trial
                    current_size = len(S)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 4. Try 2-for-3 swaps (sampled)
            s_list2 = list(S)
            random.shuffle(s_list2)
            max_pairs = min(len(s_list2) * (len(s_list2) - 1) // 2, 300)
            pairs_tried = 0
            for i in range(len(s_list2)):
                if pairs_tried >= max_pairs or time.time() > deadline - 5:
                    break
                for j in range(i + 1, len(s_list2)):
                    if pairs_tried >= max_pairs or time.time() > deadline - 5:
                        break
                    pairs_tried += 1
                    S_trial = S - {s_list2[i], s_list2[j]}
                    added = 0
                    for y in range(1, n + 1):
                        if y not in S_trial:
                            ok = True
                            for (a, b) in bad_pairs_list[y]:
                                if a in S_trial and b in S_trial:
                                    ok = False
                                    break
                            if ok:
                                S_trial.add(y)
                                added += 1
                                if added >= 3:
                                    break
                    if len(S_trial) > current_size:
                        S = S_trial
                        current_size = len(S)
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
    degrees = [len(bad_pairs_list[k]) for k in range(n + 1)]
    
    # Run multiple randomized greedy passes
    for i in range(500):
        if time.time() > deadline - 20:
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
            # Sort by degree (descending)
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
    
    # Final polish on best_set with more iterations
    best_set = local_search(best_set, max_iterations=50)
    
    return list(best_set)
