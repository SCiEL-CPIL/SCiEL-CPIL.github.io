import random
import time
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Precompute gcds
    gcds = [[0]*129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            gcds[i][j] = gcd(i, j)
            gcds[j][i] = gcd(i, j)

    def is_valid_addition(x, S_set, S_list):
        # S_set is a set for O(1) lookup if needed, S_list for iteration
        for a, b in combinations(S_list, 2):
            if gcds[a][b] == gcds[a][x] == gcds[b][x]:
                return False
        return True

    def greedy_random():
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = []
        S_set = set()
        for x in candidates:
            if is_valid_addition(x, S_set, S):
                S.append(x)
                S_set.add(x)
        return S

    best_set = greedy_random()
    
    # Local Search
    current_set = list(best_set)
    
    while time.time() - start_time < time_limit:
        # Try to add
        complement = [x for x in range(1, 129) if x not in set(current_set)]
        random.shuffle(complement)
        
        added = False
        for x in complement:
            if is_valid_addition(x, set(current_set), current_set):
                current_set.append(x)
                added = True
                if len(current_set) > len(best_set):
                    best_set = list(current_set)
                break
        
        if not added:
            # Try to move: remove one, add one
            if current_set:
                # Try a few random removals
                for _ in range(10):
                    if time.time() - start_time > time_limit: break
                    idx = random.randint(0, len(current_set) - 1)
                    orig_len = len(current_set)
                    removed = current_set.pop(idx)
                    
                    # Try to add something
                    comp = [x for x in range(1, 129) if x not in set(current_set)]
                    random.shuffle(comp)
                    found_add = False
                    for x in comp:
                        if is_valid_addition(x, set(current_set), current_set):
                            current_set.append(x)
                            found_add = True
                            break
                    
                    if found_add and len(current_set) > len(best_set):
                        best_set = list(current_set)
                    elif not found_add:
                        # Revert if we didn't grow? Or keep exploring?
                        # If we didn't grow, we are at same size or smaller.
                        # If smaller, revert to previous state?
                        # Let's just revert to best_set if we didn't improve.
                        current_set = list(best_set)
                        break
                    else:
                        # We grew current_set but not best_set? Impossible if found_add implies len+1.
                        # Unless we removed 1 and added 1, len same.
                        # If len same, we might be in a different local optimum.
                        # Keep current_set to explore further.
                        pass

    return best_set
