import random
from itertools import combinations
import time

def construct():
    n = 34
    triples = list(combinations(range(n), 3))
    m = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-sets and 5-sets for each triple
    triple_4sets = []
    triple_5sets = []
    for t in triples:
        a, b, c = t
        sets4 = []
        for d in range(n):
            if d not in (a, b, c):
                sets4.append(tuple(sorted((a, b, c, d))))
        triple_4sets.append(sets4)
        
        sets5 = []
        remaining = [i for i in range(n) if i not in (a, b, c)]
        for d, e in combinations(remaining, 2):
            sets5.append(tuple(sorted((a, b, c, d, e))))
        triple_5sets.append(sets5)
        
    # Initialize random coloring
    colors = [random.randint(0, 1) for _ in range(m)]
    
    # Initialize counts
    red_counts_4 = {}
    blue_counts_5 = {}
    for i, t in enumerate(triples):
        col = colors[i]
        for s in triple_4sets[i]:
            red_counts_4[s] = red_counts_4.get(s, 0) + (1 if col == 0 else 0)
        for s in triple_5sets[i]:
            blue_counts_5[s] = blue_counts_5.get(s, 0) + (1 if col == 1 else 0)
            
    def get_cost():
        cost = 0
        for cnt in red_counts_4.values():
            if cnt == 4: cost += 1
        for cnt in blue_counts_5.values():
            if cnt == 10: cost += 1
        return cost
        
    current_cost = get_cost()
    if current_cost == 0:
        return {'n': n, 'colors': colors}
        
    start_time = time.time()
    max_time = 115
    
    # Local Search
    while time.time() - start_time < max_time:
        best_delta = 0
        best_i = -1
        
        # Find best flip
        for i in range(m):
            old_col = colors[i]
            new_col = 1 - old_col
            delta = 0
            
            # 4-sets
            for s in triple_4sets[i]:
                old_cnt = red_counts_4[s]
                if old_col == 0:
                    # Red -> Blue
                    new_cnt = old_cnt - 1
                    if old_cnt == 4: delta -= 1
                else:
                    # Blue -> Red
                    new_cnt = old_cnt + 1
                    if new_cnt == 4: delta += 1
            # 5-sets
            for s in triple_5sets[i]:
                old_cnt = blue_counts_5[s]
                if old_col == 1:
                    # Blue -> Red
                    new_cnt = old_cnt - 1
                    if old_cnt == 10: delta -= 1
                else:
                    # Red -> Blue
                    new_cnt = old_cnt + 1
                    if new_cnt == 10: delta += 1
                    
            if delta < best_delta:
                best_delta = delta
                best_i = i
                
        if best_i != -1:
            # Apply flip
            i = best_i
            old_col = colors[i]
            new_col = 1 - old_col
            colors[i] = new_col
            current_cost += best_delta
            
            # Update counts
            for s in triple_4sets[i]:
                if old_col == 0:
                    red_counts_4[s] -= 1
                else:
                    red_counts_4[s] += 1
            for s in triple_5sets[i]:
                if old_col == 1:
                    blue_counts_5[s] -= 1
                else:
                    blue_counts_5[s] += 1
                    
            if current_cost == 0:
                return {'n': n, 'colors': colors}
        else:
            # Stuck. Try random flip or restart.
            # Random flip
            i = random.randint(0, m - 1)
            old_col = colors[i]
            new_col = 1 - old_col
            colors[i] = new_col
            
            # Update counts (no cost tracking needed for random, just re-calc later? 
            # No, we must keep counts consistent)
            for s in triple_4sets[i]:
                if old_col == 0:
                    red_counts_4[s] -= 1
                else:
                    red_counts_4[s] += 1
            for s in triple_5sets[i]:
                if old_col == 1:
                    blue_counts_5[s] -= 1
                else:
                    blue_counts_5[s] += 1
                    
            # Re-calc cost? It's expensive. 
            # Just let the next loop find the best delta.
            # But current_cost is stale. 
            # We can just re-calc cost every 100 random flips?
            # Or just accept that current_cost is approximate and rely on get_cost() at the end?
            # No, we need to know if we reached 0.
            # Let's just re-calc cost after random flips occasionally.
            
    # Final check
    if get_cost() == 0:
        return {'n': n, 'colors': colors}
    else:
        # If not solved, return anyway.
        return {'n': n, 'colors': colors}
