import random
from fractions import Fraction
import time

def construct():
    # Denominators are 2 to 48
    # Indices 0 to 46 correspond to denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    best_deficit = Fraction(1, 1) # 1.0 is max possible deficit (empty set)
    best_indices = []
    
    def get_deficit(indices):
        s = Fraction(0)
        for i in indices:
            s += Fraction(1, denominators[i])
        if s >= 1:
            return None
        return 1 - s

    def try_update(indices):
        nonlocal best_deficit, best_indices
        d = get_deficit(indices)
        if d is not None and d < best_deficit:
            best_deficit = d
            best_indices = list(indices)

    # Initial good heuristic: The set {2, 3, 7, 43}
    # Indices: 0, 1, 5, 41
    try_update([0, 1, 5, 41])
    
    # Time limit guard
    start_time = time.time()
    time_limit = 60.0 # Leave 30s margin for 90s limit
    
    # Local Search / Randomized Hill Climbing
    # We will perform several restarts with random greedy initializations
    # and then refine with local moves.
    
    for restart in range(1000): # Max restarts, but controlled by time
        if time.time() - start_time > time_limit:
            break
            
        # Generate a random permutation of indices
        perm = list(range(n))
        random.shuffle(perm)
        
        current_indices = []
        current_sum = Fraction(0)
        
        # Greedy construction
        for i in perm:
            val = Fraction(1, denominators[i])
            if current_sum + val < 1:
                current_indices.append(i)
                current_sum += val
        
        try_update(current_indices)
        
        # Local Search: Try to improve current_indices
        # We track the set as a boolean array or set for fast lookup
        # But for small N, list is fine.
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            current_set = set(current_indices)
            
            # 1. Try adding a single unused element
            for i in range(n):
                if i in current_set:
                    continue
                val = Fraction(1, denominators[i])
                if current_sum + val < 1:
                    new_indices = current_indices + [i]
                    new_sum = current_sum + val
                    new_def = 1 - new_sum
                    old_def = 1 - current_sum
                    if new_def < old_def:
                        current_indices = new_indices
                        current_sum = new_sum
                        current_set.add(i)
                        try_update(current_indices)
                        improved = True
                        break # Restart loop with new state? Or continue?
                        # Breaking ensures we re-evaluate all moves from new state
                        # But might be slow. Let's just update and continue scanning?
                        # Usually hill climbing updates and restarts the scan to avoid local traps of ordering.
            
            # 2. Try swapping a used element with an unused one
            # This is O(N^2). N=47, so 47*47 ~ 2200 checks. Fast.
            for i in range(n):
                if i not in current_set:
                    continue
                for j in range(n):
                    if j in current_set:
                        continue
                    # Remove i, add j
                    val_i = Fraction(1, denominators[i])
                    val_j = Fraction(1, denominators[j])
                    new_sum = current_sum - val_i + val_j
                    if new_sum < 1:
                        new_def = 1 - new_sum
                        old_def = 1 - current_sum
                        if new_def < old_def:
                            current_indices = [j if x == i else x for x in current_indices]
                            current_sum = new_sum
                            current_set.remove(i)
                            current_set.add(j)
                            try_update(current_indices)
                            improved = True
                            break
                if improved:
                    break
            
            # 3. Try removing one and adding two (2-for-1 swap)
            # This is O(N^3) roughly. 47^3 ~ 100k. Might be slow in Python if done many times.
            # We'll limit this or only do it if time permits.
            if time.time() - start_time < time_limit - 5:
                for i in range(n):
                    if i not in current_set:
                        continue
                    val_i = Fraction(1, denominators[i])
                    base_sum = current_sum - val_i
                    # Try adding two distinct j, k
                    # To optimize, we can iterate j, then k
                    for j in range(n):
                        if j in current_set or j == i:
                            continue
                        val_j = Fraction(1, denominators[j])
                        if base_sum + val_j >= 1:
                            continue
                        for k in range(n):
                            if k in current_set or k == i or k == j:
                                continue
                            val_k = Fraction(1, denominators[k])
                            new_sum = base_sum + val_j + val_k
                            if new_sum < 1:
                                new_def = 1 - new_sum
                                old_def = 1 - current_sum
                                if new_def < old_def:
                                    # Update
                                    new_indices = [x for x in current_indices if x != i]
                                    new_indices.append(j)
                                    new_indices.append(k)
                                    current_indices = new_indices
                                    current_sum = new_sum
                                    current_set.remove(i)
                                    current_set.add(j)
                                    current_set.add(k)
                                    try_update(current_indices)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                        # Break out of i loop? Yes, restart local search.

    return best_indices
