import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_size = 0
    best_set = set()
    
    # We will run several restarts
    # To ensure we finish, we check time frequently
    
    def run_one_restart(time_budget):
        S = np.array([], dtype=int)
        S_set = set()
        counts = np.zeros(2001, dtype=int)
        
        # Greedy Init
        for x in range(1, 1001):
            if time.time() > start_time + time_budget:
                break
            # Check if x can be added
            if len(S) == 0:
                sums = np.array([2*x])
            else:
                sums = np.empty(len(S)+1, dtype=int)
                sums[:-1] = S + x
                sums[-1] = 2*x
            
            if np.max(counts[sums]) < 2:
                counts[sums] += 1
                S = np.append(S, x)
                S = np.sort(S)
                S_set.add(x)
        
        # Local Search
        improved = True
        while improved and time.time() < start_time + time_budget:
            improved = False
            added = False
            
            # Try to add
            # Randomize order to avoid bias?
            # Let's just scan 1..1000
            for x in range(1, 1001):
                if x in S_set:
                    continue
                if len(S) == 0:
                    sums = np.array([2*x])
                else:
                    sums = np.empty(len(S)+1, dtype=int)
                    sums[:-1] = S + x
                    sums[-1] = 2*x
                
                if np.max(counts[sums]) < 2:
                    counts[sums] += 1
                    S = np.append(S, x)
                    S = np.sort(S)
                    S_set.add(x)
                    added = True
                    improved = True
                    break
            
            if not added:
                # Swap
                if len(S) > 0:
                    # Pick random element
                    idx = np.random.randint(len(S))
                    a = S[idx]
                    
                    # Remove a
                    # sums involving a
                    if len(S) > 1:
                         # S + a includes a+a
                         sums_remove = S + a
                    else:
                         sums_remove = np.array([2*a])
                    
                    counts[sums_remove] -= 1
                    S = np.delete(S, idx)
                    S_set.remove(a)
                    
                    # Try to add
                    found_b = False
                    for x in range(1, 1001):
                        if x == a or x in S_set:
                            continue
                        if len(S) == 0:
                            sums_add = np.array([2*x])
                        else:
                            sums_add = np.empty(len(S)+1, dtype=int)
                            sums_add[:-1] = S + x
                            sums_add[-1] = 2*x
                        
                        if np.max(counts[sums_add]) < 2:
                            counts[sums_add] += 1
                            S = np.append(S, x)
                            S = np.sort(S)
                            S_set.add(x)
                            found_b = True
                            improved = True
                            break
                    
                    if not found_b:
                        # Restore a
                        if len(S) == 0:
                            sums_restore = np.array([2*a])
                        else:
                            sums_restore = np.empty(len(S)+1, dtype=int)
                            sums_restore[:-1] = S + a
                            sums_restore[-1] = 2*a
                        
                        counts[sums_restore] += 1
                        S = np.append(S, a)
                        S = np.sort(S)
                        S_set.add(a)
        
        return len(S), S_set

    # Strategy: 
    # 1. Run one long restart with most of the time.
    # 2. If time remains, run short restarts.
    
    # Let's allocate 80s to the first restart, and 10s each to next 3 restarts.
    # Total 110s.
    
    budgets = [80, 10, 10, 10]
    
    for budget in budgets:
        if time.time() > deadline:
            break
        size, S_set = run_one_restart(budget)
        if size > best_size:
            best_size = size
            best_set = S_set
            
    return list(best_set)
