import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_n = 9
    best_colors = None
    
    # We will try to find a valid coloring for n in a specific range.
    # Given the difficulty, let's aim for n=20-25.
    # We'll try n=25 first. If it fails, we try 24, etc.
    # Actually, let's just iterate and keep the best.
    
    for n in range(15, 31): # Start from 15 to save time on trivial small n
        if time.time() - start_time > time_limit - 10:
            break
            
        triples = list(itertools.combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        M = len(triples)
        
        # Precompute neighbors for incremental updates
        # For each triple i, store:
        # - list of 4-sets (indices of other 3 triples)
        # - list of 5-sets (indices of other 9 triples)
        
        # This precomputation is O(n^4) + O(n^5).
        # For n=30, n^5 is 24M. This might be too slow to build for every n.
        # Let's only do this for the specific n we are testing.
        # If n=25, n^5 is ~10M. Building lists of lists might take a few seconds.
        # We have 115s.
        
        try:
            # Build data structures
            # 4-sets: for each triple, which 4-sets contain it?
            # A 4-set is defined by 4 vertices. It contains 4 triples.
            # Iterate all 4-sets.
            four_sets_per_triple = [[] for _ in range(M)]
            for quad in itertools.combinations(range(n), 4):
                tris = [triple_to_idx[t] for t in itertools.combinations(quad, 3)]
                # For each triple in this 4-set, add the other 3
                for i in range(4):
                    others = [tris[j] for j in range(4) if j != i]
                    four_sets_per_triple[tris[i]].append(others)
            
            five_sets_per_triple = [[] for _ in range(M)]
            for quint in itertools.combinations(range(n), 5):
                tris = [triple_to_idx[t] for t in itertools.combinations(quint, 3)]
                # 10 triples
                for i in range(10):
                    others = [tris[j] for j in range(10) if j != i]
                    five_sets_per_triple[tris[i]].append(others)
                    
        except Exception:
            continue

        # Initialize coloring
        C = np.random.randint(0, 2, M).astype(np.int8)
        
        # Initial conflict counts
        red_k4_count = 0
        blue_k5_count = 0
        
        # Count initial conflicts
        # We can just iterate through the stored lists? No, that's double counting.
        # Better: Iterate all 4-sets and 5-sets once.
        # But we didn't store the full list of 4-sets, only per-triple.
        # Let's just use the vectorized check for the initial count, or re-derive.
        # Actually, we can just count by checking each 4-set.
        # Let's store the 4-sets and 5-sets as lists of indices for initial count.
        
        all_four_sets = []
        for quad in itertools.combinations(range(n), 4):
            tris = [triple_to_idx[t] for t in itertools.combinations(quad, 3)]
            all_four_sets.append(tris)
            
        all_five_sets = []
        for quint in itertools.combinations(range(n), 5):
            tris = [triple_to_idx[t] for t in itertools.combinations(quint, 3)]
            all_five_sets.append(tris)
            
        for tris in all_four_sets:
            if all(C[t] == 0 for t in tris):
                red_k4_count += 1
                
        for tris in all_five_sets:
            if all(C[t] == 1 for t in tris):
                blue_k5_count += 1
                
        total_conflicts = red_k4_count + blue_k5_count
        
        found = False
        max_steps = 100000
        step = 0
        
        while step < max_steps and not found:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Pick random triple
            t_idx = np.random.randint(M)
            old_val = C[t_idx]
            new_val = 1 - old_val
            
            # Calculate change in conflicts
            delta_red = 0
            delta_blue = 0
            
            # Check 4-sets containing t_idx
            for others in four_sets_per_triple[t_idx]:
                # others is list of 3 indices
                # Check if they are all red (0)
                if all(C[t] == 0 for t in others):
                    if old_val == 0:
                        # Was red K4, now not
                        delta_red -= 1
                    else:
                        # Was not red K4 (t was blue), now is
                        delta_red += 1
                        
            # Check 5-sets containing t_idx
            for others in five_sets_per_triple[t_idx]:
                # others is list of 9 indices
                # Check if they are all blue (1)
                if all(C[t] == 1 for t in others):
                    if old_val == 1:
                        # Was blue K5, now not
                        delta_blue -= 1
                    else:
                        # Was not blue K5 (t was red), now is
                        delta_blue += 1
                        
            new_total = total_conflicts + delta_red + delta_blue
            
            # Accept if better, or with some probability if worse (SA)
            if new_total < total_conflicts:
                C[t_idx] = new_val
                red_k4_count += delta_red
                blue_k5_count += delta_blue
                total_conflicts = new_total
                if total_conflicts == 0:
                    found = True
                    best_n = n
                    best_colors = C.tolist()
                    break
            else:
                # Simple random acceptance to escape local minima
                if np.random.rand() < 0.05:
                    C[t_idx] = new_val
                    red_k4_count += delta_red
                    blue_k5_count += delta_blue
                    total_conflicts = new_total
                    if total_conflicts == 0:
                        found = True
                        best_n = n
                        best_colors = C.tolist()
                        break
                # Else revert (do nothing)
            
            step += 1
            
        if found:
            # We found a valid coloring for n.
            # We can break if n is high enough, or continue to try higher n.
            # Since we are iterating upwards, we will try n+1 next.
            pass

    if best_colors is None:
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
        
    return {'n': best_n, 'colors': best_colors}
