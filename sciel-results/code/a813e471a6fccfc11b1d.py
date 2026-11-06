import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute bad pairs for each element
    # bad_pairs_for[x] contains pairs (u, v) with u < v such that {x, u, v} is a bad triple
    bad_pairs_for = [set() for _ in range(n + 1)]
    
    # Precompute all bad triples
    # A triple {a, b, c} is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # This is equivalent to:
    # Let L = lcm(a,b). Then lcm(a,c) == L and lcm(b,c) == L.
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    # Triple (a, b, c) is bad
                    bad_pairs_for[a].add((b, c))
                    bad_pairs_for[b].add((a, c))
                    bad_pairs_for[c].add((a, b))
    
    def can_add(x, current_set):
        # Check if x can be added to current_set
        # x conflicts with current_set if there exists (u, v) in bad_pairs_for[x]
        # such that u in current_set and v in current_set
        for (u, v) in bad_pairs_for[x]:
            if u in current_set and v in current_set:
                return False
        return True

    def greedy(order):
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected

    def local_search(S):
        improved = True
        S = set(S)
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add any element
            for x in range(1, n + 1):
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if not improved:
                # 2. Try 1-for-2 swap
                # Remove x, add y, z
                found_swap = False
                for x in list(S):
                    S_trial = S - {x}
                    # Try to add two elements
                    added = []
                    for y in range(1, n + 1):
                        if y not in S_trial and can_add(y, S_trial):
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
                # 3. Try 2-for-3 swap
                # Remove x, y, add z, w, v
                found_swap = False
                s_list = list(S)
                for i in range(len(s_list)):
                    if time.time() > deadline - 2: break
                    for j in range(i + 1, len(s_list)):
                        S_trial = S - {s_list[i], s_list[j]}
                        added = []
                        for y in range(1, n + 1):
                            if y not in S_trial and can_add(y, S_trial):
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
            
            if not improved:
                # 4. Try 1-for-1 swap to escape local minima (plateau hopping)
                # This doesn't increase size, but might allow future increases
                # We do this with a probability or if stuck for a while
                # For simplicity, we'll just try a few random swaps
                found_swap = False
                for _ in range(10): # Limit attempts
                    x = random.choice(list(S))
                    S_trial = S - {x}
                    candidates = [y for y in range(1, n + 1) if y not in S_trial and can_add(y, S_trial)]
                    if candidates:
                        y = random.choice(candidates)
                        S_trial.add(y)
                        # Accept if it doesn't reduce size (it doesn't) 
                        # We accept to explore
                        S = S_trial
                        found_swap = True
                        break
                if found_swap:
                    # We made a swap, so we should continue searching
                    # But we didn't improve size. 
                    # To prevent infinite loops, we should be careful.
                    # Let's just continue the loop. The "improved" flag is False, 
                    # so the while loop will exit unless we set it True.
                    # But we didn't improve size. 
                    # Standard hill climbing would stop. 
                    # Let's not do plateau hopping to keep it simple and fast.
                    pass
        
        return S

    best_set = set()
    best_size = 0
    
    # Run multiple randomized greedy passes
    num_restarts = 200 # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() > deadline - 10: # Leave time for final local search on best
            break
            
        # Randomized order
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        # Mix in deterministic biases
        if i % 3 == 0:
             # Random
             pass
        elif i % 3 == 1:
             # Sort by number of bad pairs (conflict degree)
             degrees = [len(bad_pairs_for[k]) for k in range(n+1)]
             # Add random noise to degree
             order = sorted(order, key=lambda x: degrees[x] + random.random())
        else:
             # Sort by value
             order = list(range(1, n+1))
             if i % 6 == 2:
                 order.reverse()
        
        S = greedy(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Final polish on best_set
    best_set = local_search(best_set)
    
    return list(best_set)
