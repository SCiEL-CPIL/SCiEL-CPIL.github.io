import time
import random
from math import gcd
from itertools import combinations

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
    
    # For each x, precompute gcd(x, y) for all y
    gcd_with = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            gcd_with[i][j] = gcds[i][j]
    
    def is_valid_addition(S, x):
        """Check if adding x to S is valid. S is a list."""
        n = len(S)
        for i in range(n):
            a = S[i]
            ga_x = gcds[a][x]
            if ga_x == 1:
                continue
            for j in range(i + 1, n):
                b = S[j]
                if gcds[a][b] == ga_x and gcds[b][x] == ga_x:
                    return False
        return True
    
    def count_conflicts(S, x):
        """Count how many pairs in S would conflict with x."""
        n = len(S)
        conflicts = 0
        for i in range(n):
            a = S[i]
            ga_x = gcds[a][x]
            if ga_x == 1:
                continue
            for j in range(i + 1, n):
                b = S[j]
                if gcds[a][b] == ga_x and gcds[b][x] == ga_x:
                    conflicts += 1
        return conflicts
    
    best_set = []
    start_time = time.time()
    deadline = start_time + 55
    
    random.seed(42)
    
    # Heuristic: order by number of distinct prime factors (fewer = safer)
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    pf_counts = {x: prime_factor_count(x) for x in elements}
    
    for run in range(300):
        if time.time() > deadline:
            break
        
        # 1. Greedy Initialization with different orderings
        if run % 3 == 0:
            # Sort by ascending prime factor count (safer elements first)
            current_elements = sorted(elements, key=lambda x: pf_counts[x])
        elif run % 3 == 1:
            # Random order
            current_elements = elements[:]
            random.shuffle(current_elements)
        else:
            # Descending prime factor count (more aggressive)
            current_elements = sorted(elements, key=lambda x: -pf_counts[x])
        
        S = []
        S_set = set()
        for x in current_elements:
            if is_valid_addition(S, x):
                S.append(x)
                S_set.add(x)
        
        if len(S) > len(best_set):
            best_set = S[:]
        
        # 2. Local Search
        improved = True
        local_iterations = 0
        while improved and time.time() < deadline and local_iterations < 50:
            improved = False
            local_iterations += 1
            
            # Try to add any element not in S
            candidates = [x for x in elements if x not in S_set]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if is_valid_addition(S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # Try 1-for-1 swap: remove u, add v
            swap_found = False
            S_shuffled = S[:]
            random.shuffle(S_shuffled)
            
            for u in S_shuffled:
                if time.time() > deadline:
                    break
                
                S_temp_list = [x for x in S if x != u]
                
                # Find v that can replace u
                for v in candidates:
                    if v in S_set:
                        continue
                    if is_valid_addition(S_temp_list, v):
                        # Check that removing u and adding v doesn't create issues
                        # (it shouldn't since we're just swapping)
                        S = S_temp_list + [v]
                        S_set = set(S)
                        swap_found = True
                        improved = True
                        break
                
                if swap_found:
                    break
            
            if swap_found:
                continue
            
            # Try 1-for-2 swap: remove u, add v, w
            swap_found = False
            S_shuffled = S[:]
            random.shuffle(S_shuffled)
            
            for u in S_shuffled:
                if time.time() > deadline:
                    break
                
                S_temp_list = [x for x in S if x != u]
                S_temp_set = S_set - {u}
                
                # Find v that is valid with S - {u}
                v_found = None
                for v in candidates:
                    if v in S_set:
                        continue
                    if is_valid_addition(S_temp_list, v):
                        v_found = v
                        break
                
                if v_found is not None:
                    S_temp2_list = S_temp_list + [v_found]
                    S_temp2_set = S_temp_set | {v_found}
                    
                    # Find w that is valid with (S - {u}) + {v}
                    w_found = None
                    for w in candidates:
                        if w == v_found or w in S_set:
                            continue
                        if is_valid_addition(S_temp2_list, w):
                            w_found = w
                            break
                    
                    if w_found is not None:
                        S = S_temp2_list + [w_found]
                        S_set = S_temp2_set | {w_found}
                        swap_found = True
                        improved = True
                        break
            
            if swap_found:
                continue
            
            # Try 2-for-3 swap: remove u, v, add w, x, y
            # This is expensive, so only try occasionally
            if local_iterations % 5 == 0:
                swap_found = False
                # Try a few random 2-for-3 swaps
                for attempt in range(5):
                    if time.time() > deadline:
                        break
                    
                    # Pick 2 random elements to remove
                    if len(S) < 2:
                        break
                    remove_pair = random.sample(S, 2)
                    u, v = remove_pair
                    
                    S_temp_list = [x for x in S if x not in remove_pair]
                    S_temp_set = S_set - set(remove_pair)
                    
                    # Try to add 3 elements
                    added_count = 0
                    new_elements = []
                    remaining_candidates = [x for x in candidates if x not in S_set]
                    random.shuffle(remaining_candidates)
                    
                    for cand in remaining_candidates:
                        if added_count >= 3:
                            break
                        if is_valid_addition(S_temp_list + new_elements, cand):
                            new_elements.append(cand)
                            added_count += 1
                    
                    if added_count == 3:
                        S = S_temp_list + new_elements
                        S_set = set(S)
                        swap_found = True
                        improved = True
                        break
            
            if not swap_found:
                break
    
        if len(S) > len(best_set):
            best_set = S[:]
    
    return best_set
