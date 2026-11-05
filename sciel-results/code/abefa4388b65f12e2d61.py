import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    # Try to find a valid coloring for n=34 down to 10
    # We allocate time based on n
    # n=34 gets 80s, n=33 gets 10s, etc.
    time_budgets = {
        34: 80.0,
        33: 10.0,
        32: 5.0,
        31: 2.0,
        30: 1.0,
    }
    
    for n in range(34, 9, -1):
        if time.time() > deadline:
            break
        budget = time_budgets.get(n, 0.5)
        # Ensure we don't exceed global deadline
        budget = min(budget, deadline - time.time())
        if budget < 0.1:
            continue
            
        try:
            result = solve_n(n, budget)
            if result is not None:
                return result
        except Exception:
            continue
            
    # Fallback to seed solution
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

def solve_n(n, time_limit):
    start = time.time()
    
    # Precompute triples
    triples = list(combinations(range(n), 3))
    M = len(triples)
    idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-sets and 5-sets
    sets4 = list(combinations(range(n), 4))
    sets5 = list(combinations(range(n), 5))
    N4 = len(sets4)
    N5 = len(sets5)
    
    # Map triple index to list of 4-set indices containing it
    triple_to_4 = [[] for _ in range(M)]
    for i4, s4 in enumerate(sets4):
        for j in range(4):
            t = tuple(sorted(s4[:j] + s4[j+1:]))
            triple_to_4[idx[t]].append(i4)
            
    # Map triple index to list of 5-set indices containing it
    triple_to_5 = [[] for _ in range(M)]
    for i5, s5 in enumerate(sets5):
        for j in range(5):
            t = tuple(sorted(s5[:j] + s5[j+1:]))
            triple_to_5[idx[t]].append(i5)
            
    # Precompute triples for each 4-set and 5-set
    s4_triples = []
    for s4 in sets4:
        ts = [idx[tuple(sorted(s4[:j] + s4[j+1:]))] for j in range(4)]
        s4_triples.append(ts)
        
    s5_triples = []
    for s5 in sets5:
        ts = [idx[t] for t in combinations(s5, 3)]
        s5_triples.append(ts)
        
    # Local search with restarts
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() - start > time_limit:
            return None
            
        # Initialize random coloring
        colors = np.random.randint(0, 2, M)
        
        # Initialize counts
        red_count_4 = np.zeros(N4, dtype=np.int8)
        blue_count_5 = np.zeros(N5, dtype=np.int8)
        
        # Calculate initial counts
        for i, t in enumerate(triples):
            c = colors[i]
            for i4 in triple_to_4[i]:
                if c == 0:
                    red_count_4[i4] += 1
            for i5 in triple_to_5[i]:
                if c == 1:
                    blue_count_5[i5] += 1
                    
        # Identify violating sets
        violating_4 = set(np.where(red_count_4 == 4)[0])
        violating_5 = set(np.where(blue_count_5 == 10)[0])
        
        if not violating_4 and not violating_5:
            return {'n': n, 'colors': colors.tolist()}
            
        # Local search loop
        # We will flip triples to reduce violations
        max_iters = 50000
        for it in range(max_iters):
            if time.time() - start > time_limit:
                return None
                
            if not violating_4 and not violating_5:
                return {'n': n, 'colors': colors.tolist()}
                
            # Pick a violation
            # Use list() for random choice, but it's O(K).
            # If K is large, this is slow.
            # We can just pick a random triple from the union of violating triples?
            # For now, let's try this.
            
            if violating_4:
                if random.random() < 0.7 or not violating_5:
                    i4 = random.choice(list(violating_4))
                    t_idx = random.choice(s4_triples[i4])
                else:
                    i5 = random.choice(list(violating_5))
                    t_idx = random.choice(s5_triples[i5])
            else:
                i5 = random.choice(list(violating_5))
                t_idx = random.choice(s5_triples[i5])
                
            # Flip t_idx
            old_color = colors[t_idx]
            new_color = 1 - old_color
            colors[t_idx] = new_color
            
            # Update counts
            for i4 in triple_to_4[t_idx]:
                if old_color == 0: # was red, now blue
                    red_count_4[i4] -= 1
                    if red_count_4[i4] == 4:
                        violating_4.add(i4)
                    else:
                        violating_4.discard(i4)
                else: # was blue, now red
                    red_count_4[i4] += 1
                    if red_count_4[i4] == 4:
                        violating_4.add(i4)
                    else:
                        violating_4.discard(i4)
                        
            for i5 in triple_to_5[t_idx]:
                if old_color == 1: # was blue, now red
                    blue_count_5[i5] -= 1
                    if blue_count_5[i5] == 10:
                        violating_5.add(i5)
                    else:
                        violating_5.discard(i5)
                else: # was red, now blue
                    blue_count_5[i5] += 1
                    if blue_count_5[i5] == 10:
                        violating_5.add(i5)
                    else:
                        violating_5.discard(i5)
                        
    return None
