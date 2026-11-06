import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Target n=34
    n = 34
    
    # Precompute triple indices
    triples = list(combinations(range(n), 3))
    M = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-sets and their constituent triples
    # 4-set (a,b,c,d) has triples (a,b,c), (a,b,d), (a,c,d), (b,c,d)
    four_sets = []
    four_set_triples = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        t1 = triple_to_idx[(a, b, c)]
        t2 = triple_to_idx[(a, b, d)]
        t3 = triple_to_idx[(a, c, d)]
        t4 = triple_to_idx[(b, c, d)]
        four_sets.append(quad)
        four_set_triples.append((t1, t2, t3, t4))
    N4 = len(four_sets)
    
    # Precompute 5-sets and their constituent triples
    # 5-set (a,b,c,d,e) has 10 triples
    five_sets = []
    five_set_triples = []
    for quint in combinations(range(n), 5):
        a, b, c, d, e = quint
        ts = [
            triple_to_idx[(a, b, c)], triple_to_idx[(a, b, d)], triple_to_idx[(a, b, e)],
            triple_to_idx[(a, c, d)], triple_to_idx[(a, c, e)], triple_to_idx[(a, d, e)],
            triple_to_idx[(b, c, d)], triple_to_idx[(b, c, e)], triple_to_idx[(b, d, e)],
            triple_to_idx[(c, d, e)]
        ]
        five_sets.append(quint)
        five_set_triples.append(ts)
    N5 = len(five_sets)
    
    # Mapping from triple index to list of 4-set indices and 5-set indices containing it
    triple_to_fours = [[] for _ in range(M)]
    triple_to_fives = [[] for _ in range(M)]
    
    for i, ts in enumerate(four_set_triples):
        for t in ts:
            triple_to_fours[t].append(i)
            
    for i, ts in enumerate(five_set_triples):
        for t in ts:
            triple_to_fives[t].append(i)

    def run_sa(max_iters=500000):
        # Initial random coloring
        colors = np.random.randint(0, 2, M)
        
        # Initialize counts
        # red_count4[i] = number of red triples in 4-set i
        red_count4 = np.zeros(N4, dtype=np.int8)
        # blue_count5[i] = number of blue triples in 5-set i
        blue_count5 = np.zeros(N5, dtype=np.int8)
        
        for i, ts in enumerate(four_set_triples):
            red_count4[i] = sum(1 for t in ts if colors[t] == 0)
            
        for i, ts in enumerate(five_set_triples):
            blue_count5[i] = sum(1 for t in ts if colors[t] == 1)
            
        # Initial violations
        viol4 = np.sum(red_count4 == 4)
        viol5 = np.sum(blue_count5 == 10)
        energy = viol4 + viol5
        
        if energy == 0:
            return colors.tolist()
            
        T = 1.0
        cooling = 0.9995
        min_T = 0.001
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            if energy == 0:
                return colors.tolist()
                
            if T < min_T:
                break
                
            # Pick a random triple to flip
            t = np.random.randint(0, M)
            
            # Calculate delta energy
            # If we flip t from 0 to 1:
            # - It affects 4-sets containing t: red_count4 decreases by 1
            # - It affects 5-sets containing t: blue_count5 increases by 1
            
            # If we flip t from 1 to 0:
            # - It affects 4-sets containing t: red_count4 increases by 1
            # - It affects 5-sets containing t: blue_count5 decreases by 1
            
            delta_energy = 0
            
            if colors[t] == 0:
                # Flip 0 -> 1
                # 4-sets: red_count4[i] -= 1
                for i in triple_to_fours[t]:
                    if red_count4[i] == 4:
                        delta_energy -= 1 # Violation removed
                    elif red_count4[i] == 3:
                        delta_energy += 0 # No new violation (3->2 is fine, 4->3 is removal)
                        # Wait, if it was 4, it becomes 3 (removal). If it was 3, it becomes 2 (no change in violation status).
                        # So only 4->3 matters for reduction.
                    # Actually, logic:
                    # Old: red_count4[i]. New: red_count4[i] - 1.
                    # Violation if count == 4.
                    # If old == 4, new == 3: Violation removed (-1)
                    # If old != 4, new != 4 (since new < old): No change.
                    # So just check if old == 4.
                    
                # 5-sets: blue_count5[i] += 1
                for i in triple_to_fives[t]:
                    if blue_count5[i] == 9:
                        delta_energy += 1 # New violation created (9->10)
                    # If old != 9, new != 10 (since new > old, so if old=10, new=11, still violation? 
                    # No, if old=10, it was already a violation. New=11 is also a violation. 
                    # The count of violations doesn't change if it goes 10->11.
                    # Only 9->10 creates a new violation.
                    # 10->11 keeps the violation count same.
                    # So only check if old == 9.
            else:
                # Flip 1 -> 0
                # 4-sets: red_count4[i] += 1
                for i in triple_to_fours[t]:
                    if red_count4[i] == 3:
                        delta_energy += 1 # New violation created (3->4)
                # 5-sets: blue_count5[i] -= 1
                for i in triple_to_fives[t]:
                    if blue_count5[i] == 10:
                        delta_energy -= 1 # Violation removed (10->9)
            
            # Acceptance
            if delta_energy < 0 or (T > 0 and np.random.rand() < np.exp(-delta_energy / T)):
                # Apply flip
                colors[t] = 1 - colors[t]
                energy += delta_energy
                
                # Update counts
                if colors[t] == 1:
                    # It was 0, now 1
                    for i in triple_to_fours[t]:
                        red_count4[i] -= 1
                    for i in triple_to_fives[t]:
                        blue_count5[i] += 1
                else:
                    # It was 1, now 0
                    for i in triple_to_fours[t]:
                        red_count4[i] += 1
                    for i in triple_to_fives[t]:
                        blue_count5[i] -= 1
                
                T *= cooling
                
        return None

    # Try multiple restarts
    best_result = None
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        result = run_sa()
        if result is not None:
            best_result = result
            break
            
    if best_result is not None:
        return {'n': n, 'colors': best_result}
    else:
        # Fallback: try smaller n if 34 fails?
        # But 34 is the target. If we fail, we score 0.
        # We can try a greedy repair on the best partial solution?
        # For now, return a valid coloring for a smaller n to ensure non-zero score if possible?
        # No, the task says "Maximize n". If we can't find 34, maybe try 30?
        # But 30 is much easier.
        # Let's try to find a valid coloring for n=30 as a fallback.
        
        def find_valid_n(n_target):
             # Simplified SA for smaller n
             triples = list(combinations(range(n_target), 3))
             M = len(triples)
             triple_to_idx = {t: i for i, t in enumerate(triples)}
             
             four_set_triples = []
             for quad in combinations(range(n_target), 4):
                 a, b, c, d = quad
                 ts = [triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)], triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]]
                 four_set_triples.append(ts)
             N4 = len(four_set_triples)
             
             five_set_triples = []
             for quint in combinations(range(n_target), 5):
                 a, b, c, d, e = quint
                 ts = [
                     triple_to_idx[(a, b, c)], triple_to_idx[(a, b, d)], triple_to_idx[(a, b, e)],
                     triple_to_idx[(a, c, d)], triple_to_idx[(a, c, e)], triple_to_idx[(a, d, e)],
                     triple_to_idx[(b, c, d)], triple_to_idx[(b, c, e)], triple_to_idx[(b, d, e)],
                     triple_to_idx[(c, d, e)]
                 ]
                 five_set_triples.append(ts)
             N5 = len(five_set_triples)
             
             triple_to_fours = [[] for _ in range(M)]
             triple_to_fives = [[] for _ in range(M)]
             for i, ts in enumerate(four_set_triples):
                 for t in ts:
                     triple_to_fours[t].append(i)
             for i, ts in enumerate(five_set_triples):
                 for t in ts:
                     triple_to_fives[t].append(i)
                     
             # SA
             colors = np.random.randint(0, 2, M)
             red_count4 = np.zeros(N4, dtype=np.int8)
             blue_count5 = np.zeros(N5, dtype=np.int8)
             for i, ts in enumerate(four_set_triples):
                 red_count4[i] = sum(1 for t in ts if colors[t] == 0)
             for i, ts in enumerate(five_set_triples):
                 blue_count5[i] = sum(1 for t in ts if colors[t] == 1)
             energy = np.sum(red_count4 == 4) + np.sum(blue_count5 == 10)
             
             T = 1.0
             cooling = 0.999
             for _ in range(100000):
                 if time.time() > deadline - 2:
                     break
                 if energy == 0:
                     return {'n': n_target, 'colors': colors.tolist()}
                 
                 t = np.random.randint(0, M)
                 delta_energy = 0
                 if colors[t] == 0:
                     for i in triple_to_fours[t]:
                         if red_count4[i] == 4: delta_energy -= 1
                     for i in triple_to_fives[t]:
                         if blue_count5[i] == 9: delta_energy += 1
                 else:
                     for i in triple_to_fours[t]:
                         if red_count4[i] == 3: delta_energy += 1
                     for i in triple_to_fives[t]:
                         if blue_count5[i] == 10: delta_energy -= 1
                 
                 if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / T):
                     colors[t] = 1 - colors[t]
                     energy += delta_energy
                     if colors[t] == 1:
                         for i in triple_to_fours[t]: red_count4[i] -= 1
                         for i in triple_to_fives[t]: blue_count5[i] += 1
                     else:
                         for i in triple_to_fours[t]: red_count4[i] += 1
                         for i in triple_to_fives[t]: blue_count5[i] -= 1
                     T *= cooling
             return None
             
        for n_fallback in [30, 25, 20]:
            if time.time() > deadline - 1:
                break
            res = find_valid_n(n_fallback)
            if res:
                return res
                
        # Last resort: return a trivially valid small n
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
