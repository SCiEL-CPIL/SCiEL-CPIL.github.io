"""B2[2] set: greedy + local search with O(n) per-operation sum counting."""
import numpy as np
import time

def construct():
    deadline = time.time() + 130
    MAX_SUM = 2000

    # Greedy construction
    chosen = []
    sum_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    
    for x in range(1, 1001):
        # Check: for all y in chosen, sum_counts[x+y] < 2, and sum_counts[2x] < 2
        ok = True
        if 2 * x <= MAX_SUM and sum_counts[2 * x] >= 2:
            ok = False
        else:
            for y in chosen:
                if sum_counts[x + y] >= 2:
                    ok = False
                    break
        if ok:
            for y in chosen:
                sum_counts[x + y] += 1
            if 2 * x <= MAX_SUM:
                sum_counts[2 * x] += 1
            chosen.append(x)
    
    # Local search: try to improve by swapping
    chosen_set = set(chosen)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        improved = False
        n = len(chosen)
        
        for idx in range(n):
            if time.time() > deadline:
                break
            
            x_rem = chosen[idx]
            
            # Remove x_rem
            for j in range(n):
                if j == idx:
                    continue
                s = x_rem + chosen[j]
                if s <= MAX_SUM:
                    sum_counts[s] -= 1
            s2 = 2 * x_rem
            if s2 <= MAX_SUM:
                sum_counts[s2] -= 1
            
            n -= 1
            chosen.pop(idx)
            chosen_set.discard(x_rem)
            
            # Try to add 2 new elements
            added = []
            for x_new in range(1, 1001):
                if time.time() > deadline:
                    break
                if x_new in chosen_set:
                    continue
                # Check validity
                ok = True
                if 2 * x_new <= MAX_SUM and sum_counts[2 * x_new] >= 2:
                    ok = False
                else:
                    for y in chosen:
                        if sum_counts[x_new + y] >= 2:
                            ok = False
                            break
                if ok:
                    for y in chosen:
                        sum_counts[x_new + y] += 1
                    if 2 * x_new <= MAX_SUM:
                        sum_counts[2 * x_new] += 1
                    chosen.append(x_new)
                    chosen_set.add(x_new)
                    added.append(x_new)
                    if len(added) == 2:
                        break
            
            if len(added) >= 2:
                improved = True
            else:
                # Undo: remove added, restore x_rem
                for x_a in added:
                    for y in chosen:
                        if y == x_a:
                            continue
                        s = x_a + y
                        if s <= MAX_SUM:
                            sum_counts[s] -= 1
                    s2a = 2 * x_a
                    if s2a <= MAX_SUM:
                        sum_counts[s2a] -= 1
                    chosen.remove(x_a)
                    chosen_set.discard(x_a)
                
                # Restore x_rem
                for y in chosen:
                    s = x_rem + y
                    if s <= MAX_SUM:
                        sum_counts[s] += 1
                s2 = 2 * x_rem
                if s2 <= MAX_SUM:
                    sum_counts[s2] += 1
                chosen.append(x_rem)
                chosen_set.add(x_rem)
        
        if not improved:
            break
    
    return chosen
