import random
import time
import itertools
import numpy as np

def construct():
    # We want to maximize n.
    # Known lower bound is 34.
    # We will attempt to find a valid coloring for n=34 using local search.
    # If that fails within time, we try smaller n.
    # Fallback is n=9.
    
    target_ns = [34, 30, 25, 20, 15, 10, 9]
    
    # Helper to check validity and return coloring
    def try_solve(n, time_limit=110):
        # Generate all triples
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map triple index to (a,b,c)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Generate all 4-sets
        quads = list(itertools.combinations(range(n), 4))
        num_quads = len(quads)
        
        # Generate all 5-sets
        quints = list(itertools.combinations(range(n), 5))
        num_quints = len(quints)
        
        # Map quad to its 4 triples
        quad_to_triples = []
        for q in quads:
            ts = list(itertools.combinations(q, 3))
            idxs = [triple_to_idx[t] for t in ts]
            quad_to_triples.append(idxs)
            
        # Map quint to its 10 triples
        quint_to_triples = []
        for q in quints:
            ts = list(itertools.combinations(q, 3))
            idxs = [triple_to_idx[t] for t in ts]
            quint_to_triples.append(idxs)
            
        # Map triple index to quads it belongs to
        triple_to_quads = [[] for _ in range(num_triples)]
        for i, ts in enumerate(quad_to_triples):
            for t_idx in ts:
                triple_to_quads[t_idx].append(i)
                
        # Map triple index to quints it belongs to
        triple_to_quints = [[] for _ in range(num_triples)]
        for i, ts in enumerate(quint_to_triples):
            for t_idx in ts:
                triple_to_quints[t_idx].append(i)
                
        # Initialize colors randomly
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Initialize counts
        # red_count_4[i] = number of red triples in quad i
        red_count_4 = np.zeros(num_quads, dtype=np.int8)
        # blue_count_5[j] = number of blue triples in quint j
        blue_count_5 = np.zeros(num_quints, dtype=np.int8)
        
        # Populate initial counts
        for t_idx, c in enumerate(colors):
            if c == 0: # Red
                for q_idx in triple_to_quads[t_idx]:
                    red_count_4[q_idx] += 1
            else: # Blue
                for q_idx in triple_to_quints[t_idx]:
                    blue_count_5[q_idx] += 1
                    
        # Identify violations
        # Violated quads: red_count_4 == 4
        # Violated quints: blue_count_5 == 10
        violated_quads = set(np.where(red_count_4 == 4)[0])
        violated_quints = set(np.where(blue_count_5 == 10)[0])
        
        start_time = time.time()
        
        while violated_quads or violated_quints:
            if time.time() - start_time > time_limit:
                return None
            
            # Try to fix a red K4
            if violated_quads:
                q_idx = random.choice(list(violated_quads))
                ts = quad_to_triples[q_idx]
                # Find a red triple in this quad
                red_ts = [t for t in ts if colors[t] == 0]
                if red_ts:
                    t_idx = random.choice(red_ts)
                    # Flip to blue (1)
                    colors[t_idx] = 1
                    
                    # Update counts
                    # This triple was red, now blue.
                    # Decrease red_count_4 for all quads containing t_idx
                    for q in triple_to_quads[t_idx]:
                        red_count_4[q] -= 1
                        if red_count_4[q] == 4:
                            violated_quads.add(q)
                        elif red_count_4[q] == 3:
                            if q in violated_quads:
                                violated_quads.remove(q)
                        
                    # Increase blue_count_5 for all quints containing t_idx
                    for q in triple_to_quints[t_idx]:
                        blue_count_5[q] += 1
                        if blue_count_5[q] == 10:
                            violated_quints.add(q)
                        elif blue_count_5[q] == 9:
                            if q in violated_quints:
                                violated_quints.remove(q)
                    
                    # Remove the fixed quad from violated list if it's no longer violated
                    if red_count_4[q_idx] < 4:
                        violated_quads.discard(q_idx)
                        
            # Try to fix a blue K5
            elif violated_quints:
                q_idx = random.choice(list(violated_quints))
                ts = quint_to_triples[q_idx]
                # Find a blue triple in this quint
                blue_ts = [t for t in ts if colors[t] == 1]
                if blue_ts:
                    t_idx = random.choice(blue_ts)
                    # Flip to red (0)
                    colors[t_idx] = 0
                    
                    # Update counts
                    # This triple was blue, now red.
                    # Decrease blue_count_5 for all quints containing t_idx
                    for q in triple_to_quints[t_idx]:
                        blue_count_5[q] -= 1
                        if blue_count_5[q] == 10:
                            violated_quints.add(q)
                        elif blue_count_5[q] == 9:
                            if q in violated_quints:
                                violated_quints.remove(q)
                        
                    # Increase red_count_4 for all quads containing t_idx
                    for q in triple_to_quads[t_idx]:
                        red_count_4[q] += 1
                        if red_count_4[q] == 4:
                            violated_quads.add(q)
                        elif red_count_4[q] == 3:
                            if q in violated_quads:
                                violated_quads.remove(q)
                    
                    # Remove the fixed quint from violated list if it's no longer violated
                    if blue_count_5[q_idx] < 10:
                        violated_quints.discard(q_idx)
                        
        if not violated_quads and not violated_quints:
            return {'n': n, 'colors': colors.tolist()}
            
        return None

    # Try to solve for larger n first
    for n in target_ns:
        res = try_solve(n, time_limit=10) # Use short time limit per attempt
        if res:
            return res
            
    # Fallback to seed solution
    from itertools import combinations
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
