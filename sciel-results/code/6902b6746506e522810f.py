import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try to find a valid coloring for n from 34 down to 10
    # We skip some values to save time if large ones fail quickly
    target_ns = [34, 32, 30, 28, 25, 22, 20, 18, 15, 12, 10]
    
    for n in target_ns:
        if time.time() > deadline - 5.0:
            break
        result = solve_n(n, deadline)
        if result is not None:
            return result
            
    # Fallback: Return a trivially valid small n
    # n=4: 4 triples. Color: 0, 1, 1, 1.
    # (0,1,2)->0, (0,1,3)->1, (0,2,3)->1, (1,2,3)->1
    # Red: {(0,1,2)}. No Red K4.
    # Blue: {(0,1,3), (0,2,3), (1,2,3)}. No Blue K5.
    n = 4
    colors = [0, 1, 1, 1]
    return {'n': n, 'colors': colors}

def solve_n(n, deadline):
    # Precompute structures
    triples = list(combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(triples)}
    num_triples = len(triples)
    
    quads = list(combinations(range(n), 4))
    quints = list(combinations(range(n), 5))
    num_quads = len(quads)
    num_quints = len(quints)
    
    # Map triple index to list of quad indices containing it
    triple_to_quads = [[] for _ in range(num_triples)]
    # Map triple index to list of quint indices containing it
    triple_to_quints = [[] for _ in range(num_triples)]
    
    # Map quad index to its 4 triple indices
    quad_to_triples = [None] * num_quads
    for q_idx, q in enumerate(quads):
        a, b, c, d = q
        t1 = triple_index[(a, b, c)]
        t2 = triple_index[(a, b, d)]
        t3 = triple_index[(a, c, d)]
        t4 = triple_index[(b, c, d)]
        quad_to_triples[q_idx] = [t1, t2, t3, t4]
        for t in [t1, t2, t3, t4]:
            triple_to_quads[t].append(q_idx)
            
    # Map quint index to its 10 triple indices
    quint_to_triples = [None] * num_quints
    for p_idx, p in enumerate(quints):
        a, b, c, d, e = p
        ts = []
        for comb in combinations([a,b,c,d,e], 3):
            ts.append(triple_index[comb])
        quint_to_triples[p_idx] = ts
        for t in ts:
            triple_to_quints[t].append(p_idx)
            
    # Try multiple restarts with structured initialization
    max_restarts = 10
    time_per_restart = (deadline - time.time()) / max_restarts
    if time_per_restart < 0.1:
        return None
        
    for restart in range(max_restarts):
        if time.time() > deadline - 1.0:
            break
            
        # Structured Initialization: Linear form mod 2
        c1 = random.randint(0, 1)
        c2 = random.randint(0, 1)
        c3 = random.randint(0, 1)
        if c1==0 and c2==0 and c3==0:
            c1 = 1
            
        colors = [0] * num_triples
        for i, (a, b, c) in enumerate(triples):
            colors[i] = (c1*a + c2*b + c3*c) % 2
            
        # Calculate initial counts
        red_quad_counts = [0] * num_quads
        blue_quint_counts = [0] * num_quints
        
        for q_idx, ts in enumerate(quad_to_triples):
            count = 0
            for t in ts:
                if colors[t] == 0:
                    count += 1
            red_quad_counts[q_idx] = count
            
        for p_idx, ts in enumerate(quint_to_triples):
            count = 0
            for t in ts:
                if colors[t] == 1:
                    count += 1
            blue_quint_counts[p_idx] = count
            
        red_violations = sum(1 for c in red_quad_counts if c == 4)
        blue_violations = sum(1 for c in blue_quint_counts if c == 10)
        
        if red_violations == 0 and blue_violations == 0:
            return {'n': n, 'colors': colors}
            
        # Local Search: Min-Conflict
        # We try to flip triples involved in violations
        max_iters = int(time_per_restart * 5000)
        if max_iters < 100:
            max_iters = 100
            
        for _ in range(max_iters):
            if time.time() > deadline - 1.0:
                break
                
            # Pick a triple to flip
            # Strategy: Find a violating quad or quint, pick a triple in it
            # If none found quickly, pick random
            i = -1
            if red_violations > 0:
                for _ in range(5):
                    q_idx = random.randint(0, num_quads - 1)
                    if red_quad_counts[q_idx] == 4:
                        i = random.choice(quad_to_triples[q_idx])
                        break
            if i == -1 and blue_violations > 0:
                for _ in range(5):
                    p_idx = random.randint(0, num_quints - 1)
                    if blue_quint_counts[p_idx] == 10:
                        i = random.choice(quint_to_triples[p_idx])
                        break
            if i == -1:
                i = random.randint(0, num_triples - 1)
                
            # Flip color i
            old_color = colors[i]
            new_color = 1 - old_color
            colors[i] = new_color
            
            # Update counts and violations
            for q_idx in triple_to_quads[i]:
                if old_color == 0: # Red -> Blue
                    if red_quad_counts[q_idx] == 4:
                        red_violations -= 1
                    red_quad_counts[q_idx] -= 1
                else: # Blue -> Red
                    if red_quad_counts[q_idx] + 1 == 4:
                        red_violations += 1
                    red_quad_counts[q_idx] += 1
                    
            for p_idx in triple_to_quints[i]:
                if old_color == 0: # Red -> Blue
                    if blue_quint_counts[p_idx] + 1 == 10:
                        blue_violations += 1
                    blue_quint_counts[p_idx] += 1
                else: # Blue -> Red
                    if blue_quint_counts[p_idx] == 10:
                        blue_violations -= 1
                    blue_quint_counts[p_idx] -= 1
                    
            if red_violations == 0 and blue_violations == 0:
                return {'n': n, 'colors': colors}
                
    return None
