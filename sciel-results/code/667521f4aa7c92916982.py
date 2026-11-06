import numpy as np
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Structures for current n
    def get_structures(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        quads = list(combinations(range(n), 4))
        quad_triples = []
        for q in quads:
            i, j, k, l = q
            quad_triples.append([triple_idx[(i, j, k)], triple_idx[(i, j, l)], triple_idx[(i, k, l)], triple_idx[(j, k, l)]])
            
        pents = list(combinations(range(n), 5))
        pent_triples = []
        for p in pents:
            i, j, k, l, m = p
            ts = [triple_idx[c] for c in combinations(p, 3)]
            pent_triples.append(ts)
            
        triple_in_quads = [[] for _ in range(num_triples)]
        for q_idx, ts in enumerate(quad_triples):
            for t in ts:
                triple_in_quads[t].append(q_idx)
                
        triple_in_pents = [[] for _ in range(num_triples)]
        for p_idx, ts in enumerate(pent_triples):
            for t in ts:
                triple_in_pents[t].append(p_idx)
                
        return triples, quad_triples, pent_triples, triple_in_quads, triple_in_pents

    def local_search(n, colors, structures, time_limit):
        triples, quad_triples, pent_triples, triple_in_quads, triple_in_pents = structures
        num_triples = len(triples)
        
        red_k4_count = np.zeros(len(quad_triples), dtype=np.int8)
        blue_k5_count = np.zeros(len(pent_triples), dtype=np.int8)
        
        for t_idx, c in enumerate(colors):
            if c == 0:
                for q_idx in triple_in_quads[t_idx]:
                    red_k4_count[q_idx] += 1
            else:
                for p_idx in triple_in_pents[t_idx]:
                    blue_k5_count[p_idx] += 1
                    
        violations = np.sum(red_k4_count == 4) + np.sum(blue_k5_count == 10)
        
        t_end = time.time() + time_limit
        while time.time() < t_end:
            if violations == 0:
                return colors, True
            
            t_idx = random.randint(0, num_triples - 1)
            c = colors[t_idx]
            
            delta_violations = 0
            
            for q_idx in triple_in_quads[t_idx]:
                old_count = red_k4_count[q_idx]
                new_count = old_count - 1 if c == 0 else old_count + 1
                if old_count == 4: delta_violations -= 1
                if new_count == 4: delta_violations += 1
            
            for p_idx in triple_in_pents[t_idx]:
                old_count = blue_k5_count[p_idx]
                new_count = old_count - 1 if c == 1 else old_count + 1
                if old_count == 10: delta_violations -= 1
                if new_count == 10: delta_violations += 1
            
            if delta_violations < 0 or (delta_violations == 0 and random.random() < 0.5):
                colors[t_idx] = 1 - c
                for q_idx in triple_in_quads[t_idx]:
                    if c == 0: red_k4_count[q_idx] -= 1
                    else: red_k4_count[q_idx] += 1
                for p_idx in triple_in_pents[t_idx]:
                    if c == 1: blue_k5_count[p_idx] -= 1
                    else: blue_k5_count[p_idx] += 1
                violations += delta_violations
                
        return colors, False

    # Start with n=20
    current_n = 20
    structures = get_structures(current_n)
    num_triples = len(structures[0])
    current_colors = np.random.randint(0, 2, size=num_triples)
    
    # Solve for n=20
    time_limit = 5.0
    current_colors, valid = local_search(current_n, current_colors, structures, time_limit)
    
    if not valid:
        # If we can't solve 20, return seed
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
        
    best_n = current_n
    best_colors = current_colors.tolist()
    
    # Try to increase n
    while current_n < 35:
        if time.time() > deadline - 5:
            break
            
        current_n += 1
        
        # Get new structures
        new_structures = get_structures(current_n)
        new_triples = new_structures[0]
        new_num_triples = len(new_triples)
        
        # Map old colors to new colors
        # Old triples are combinations of range(current_n-1)
        old_triples = list(combinations(range(current_n-1), 3))
        
        # Create dict for new triples
        new_triple_dict = {t: i for i, t in enumerate(new_triples)}
        
        new_colors = np.random.randint(0, 2, size=new_num_triples)
        
        # Copy old colors
        for i, t in enumerate(old_triples):
            # t is a tuple from old set, which is also in new set
            new_idx = new_triple_dict[t]
            new_colors[new_idx] = current_colors[i]
            
        # Run local search with remaining time
        time_left = deadline - time.time() - 2
        if time_left < 1:
            break
            
        new_colors, valid = local_search(current_n, new_colors, new_structures, time_left)
        
        if valid:
            best_n = current_n
            best_colors = new_colors.tolist()
            current_colors = new_colors
            # Continue to next n
        else:
            # If we fail, we can't guarantee we can solve larger n
            # So we stop
            break
            
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
