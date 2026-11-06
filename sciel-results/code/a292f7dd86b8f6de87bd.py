import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_n = 9
    best_colors = None
    
    # Try n from 25 down to 10? Or up?
    # Let's try to find the max n we can solve.
    # We'll iterate n from 25 down to 10, and return the first valid one found?
    # No, we want to maximize n. So we should try high n first.
    # But high n is harder.
    # Strategy: Try n=25. If found, return. If not, try n=24, etc.
    # This ensures we return a valid high n if possible.
    
    for n in [25, 24, 23, 22, 21, 20, 18, 15, 10]:
        if time.time() - start_time > time_limit - 10:
            break
            
        triples = list(itertools.combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        M = len(triples)
        
        # Precompute neighbors
        four_sets_per_triple = [[] for _ in range(M)]
        for quad in itertools.combinations(range(n), 4):
            tris = [triple_to_idx[t] for t in itertools.combinations(quad, 3)]
            for i in range(4):
                others = [tris[j] for j in range(4) if j != i]
                four_sets_per_triple[tris[i]].append(others)
                
        five_sets_per_triple = [[] for _ in range(M)]
        for quint in itertools.combinations(range(n), 5):
            tris = [triple_to_idx[t] for t in itertools.combinations(quint, 3)]
            for i in range(10):
                others = [tris[j] for j in range(10) if j != i]
                five_sets_per_triple[tris[i]].append(others)
                
        # Convert neighbor lists to numpy arrays for faster access?
        # Or just use them as lists.
        
        # Initialize coloring
        C = np.random.randint(0, 2, M).astype(np.int8)
        
        # Initial conflict count
        # We can compute this efficiently by iterating all 4-sets and 5-sets
        red_k4_count = 0
        blue_k5_count = 0
        
        for quad in itertools.combinations(range(n), 4):
            tris = [triple_to_idx[t] for t in itertools.combinations(quad, 3)]
            if all(C[t] == 0 for t in tris):
                red_k4_count += 1
                
        for quint in itertools.combinations(range(n), 5):
            tris = [triple_to_idx[t] for t in itertools.combinations(quint, 3)]
            if all(C[t] == 1 for t in tris):
                blue_k5_count += 1
                
        total_conflicts = red_k4_count + blue_k5_count
        
        found = False
        max_steps = 50000 # Reduced steps to ensure time limit
        
        while max_steps > 0 and not found:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Pick random triple
            t_idx = np.random.randint(M)
            old_val = C[t_idx]
            new_val = 1 - old_val
            
            delta_red = 0
            delta_blue = 0
            
            # Check 4-sets
            for others in four_sets_per_triple[t_idx]:
                # others is a list of 3 indices
                # Use numpy for speed
                if np.all(C[others] == 0):
                    if old_val == 0:
                        delta_red -= 1
                    else:
                        delta_red += 1
                        
            # Check 5-sets
            for others in five_sets_per_triple[t_idx]:
                # others is a list of 9 indices
                if np.all(C[others] == 1):
                    if old_val == 1:
                        delta_blue -= 1
                    else:
                        delta_blue += 1
                        
            new_total = total_conflicts + delta_red + delta_blue
            
            # Acceptance criterion
            # Simple: accept if better, or with small prob if worse
            if new_total < total_conflicts or (np.random.rand() < 0.05):
                C[t_idx] = new_val
                red_k4_count += delta_red
                blue_k5_count += delta_blue
                total_conflicts = new_total
                
                if total_conflicts == 0:
                    found = True
                    best_n = n
                    best_colors = C.tolist()
                    break
                    
            max_steps -= 1
            
        if found:
            break # Found a valid coloring for this n, which is the highest we tried

    if best_colors is None:
        # Fallback to a small valid coloring
        n = 9
        # A simple valid coloring for n=9?
        # Random might fail. Let's use a deterministic one or just return n=1.
        # Actually, n=1 is always valid.
        # But we want to maximize score.
        # If we didn't find anything, return n=1.
        return {'n': 1, 'colors': []}
        
    return {'n': best_n, 'colors': best_colors}
