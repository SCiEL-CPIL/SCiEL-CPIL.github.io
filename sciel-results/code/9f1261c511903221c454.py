import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 55.0 # 5s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    values = [Fraction(1, d) for d in denominators]
    
    best_indices = []
    best_sum = Fraction(0)
    
    # 1. Greedy Initialization
    def greedy_solve(order):
        s = Fraction(0)
        inds = []
        for i in order:
            if s + values[i] < 1:
                s += values[i]
                inds.append(i)
        return inds, s
        
    # Initial best from standard greedy (smallest denominators first)
    order = list(range(n))
    best_indices, best_sum = greedy_solve(order)
    
    # Try some random greedy starts to get a better initial point
    for _ in range(2000):
        if time.time() - start_time > time_limit: break
        random.shuffle(order)
        inds, s = greedy_solve(order)
        if s > best_sum:
            best_sum = s
            best_indices = inds
    
    # Local Search
    best_set = set(best_indices)
    
    # Helper to calculate sum from set (for verification if needed, but we track it)
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        # Try all single swaps
        for i in list(best_set):
            val_i = values[i]
            base_sum = best_sum - val_i
            for j in range(n):
                if j in best_set: continue
                val_j = values[j]
                candidate_sum = base_sum + val_j
                if candidate_sum < 1:
                    if candidate_sum > best_sum:
                        best_sum = candidate_sum
                        best_set.remove(i)
                        best_set.add(j)
                        improved = True
                        break
            if improved:
                break
                
    # If time remains, try 2-swap moves?
    # 2-swap: remove i, k; add j, l
    # This is O(N^4) which is 47^4 ~ 5M. Too slow for full sweep in Python in 5s.
    # But we can do random 2-swaps.
    
    if time.time() - start_time < time_limit:
        # Random 2-swap local search
        for _ in range(5000):
            if time.time() - start_time > time_limit: break
            # Pick 2 to remove
            if len(best_set) < 2: break
            rem_items = list(best_set)
            random.shuffle(rem_items)
            i, k = rem_items[0], rem_items[1]
            
            # Pick 2 to add
            avail = [x for x in range(n) if x not in best_set]
            if len(avail) < 2: break
            random.shuffle(avail)
            j, l = avail[0], avail[1]
            
            candidate_sum = best_sum - values[i] - values[k] + values[j] + values[l]
            if candidate_sum < 1 and candidate_sum > best_sum:
                best_sum = candidate_sum
                best_set.remove(i)
                best_set.remove(k)
                best_set.add(j)
                best_set.add(l)
                
    return list(best_set)
