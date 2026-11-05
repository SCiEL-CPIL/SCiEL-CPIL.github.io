import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    # Precompute combinations for efficiency if needed, 
    # but for n<=40, on-the-fly is fine.
    
    best_n = 0
    best_colors = None
    
    # Try to find valid colorings for increasing n
    # We know n=34 is possible. Let's try up to 34.
    # If we fail on 34, we might have a valid one for smaller n from previous attempts?
    # No, a valid coloring for n doesn't imply one for n+1.
    # We should search for the largest n.
    
    # Start from high n and go down if we can't solve it quickly?
    # Or start from a known good structure?
    
    # Let's try a few values of n.
    # The prompt mentions 34 is the published construction.
    # We will attempt to find a solution for n=34.
    
    for target_n in [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 15, 10, 9]:
        if time.time() > deadline - 20:
            break
            
        if target_n <= best_n:
            continue
            
        # Try to solve for target_n
        result = solve_n(target_n, deadline - 5)
        if result is not None:
            best_n = target_n
            best_colors = result
            # If we found 34, we can stop or try 35?
            # R(4,5;3) is known to be > 34. Is it > 35?
            # Literature: R(4,5;3) is between 35 and 63? No, 
            # Actually, let's just stick to finding 34 first.
            # If we find 34, we can try 35 in remaining time.
            if target_n == 34:
                # Try 35 if time permits
                if time.time() < deadline - 10:
                    result_35 = solve_n(35, deadline - 5)
                    if result_35 is not None:
                        best_n = 35
                        best_colors = result_35
                break
                
    if best_colors is None:
        # Fallback to seed
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
        
    return {'n': best_n, 'colors': best_colors}

def solve_n(n, deadline):
    # Initialize colors
    # Use a heuristic initialization to get close to a solution
    # e.g. based on sum of indices mod 3 or 4
    
    # Let's try a few initializations
    inits = [
        lambda a,b,c: (a+b+c) % 2,
        lambda a,b,c: (a+b) % 3,
        lambda a,b,c: (a*b+c) % 2,
        lambda a,b,c: 0, # All red (bad for K4, good for K5? No, all red means many K4s)
        lambda a,b,c: 1, # All blue (many K5s)
    ]
    
    # Random initialization is also good
    np.random.seed(42)
    
    best_score = float('inf')
    best_config = None
    
    # Precompute list of triples
    triples = list(itertools.combinations(range(n), 3))
    triple_idx = {t: i for i, t in enumerate(triples)}
    num_triples = len(triples)
    
    # Precompute for each triple, the 4-sets and 5-sets it belongs to
    # 4-sets containing (a,b,c): {a,b,c,d} for d in range(n) if d not in (a,b,c)
    # 5-sets containing (a,b,c): {a,b,c,d,e} for d<e, d,e not in (a,b,c)
    
    # To speed up violation counting, we can precompute the "neighbors"
    # But let's just iterate.
    
    for init_func in inits:
        colors = np.array([init_func(a,b,c) for a,b,c in triples], dtype=np.int8)
        
        # Calculate initial violations
        v4, v5 = count_violations(colors, n, triples, triple_idx)
        score = v4 + v5
        
        if score < best_score:
            best_score = score
            best_config = colors.copy()
            
        if score == 0:
            return colors.tolist()
            
        # Local Search
        T = 1.0 # Temperature
        for step in range(5000):
            if time.time() > deadline:
                break
                
            # Pick random triple
            idx = np.random.randint(num_triples)
            
            # Calculate delta
            delta = calc_delta(colors, idx, n, triples, triple_idx)
            
            # Metropolis criterion
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                colors[idx] = 1 - colors[idx]
                score += delta
                if score < best_score:
                    best_score = score
                    best_config = colors.copy()
                    if score == 0:
                        return colors.tolist()
                        
            # Anneal
            T *= 0.99
            
    if best_config is not None and best_score == 0:
        return best_config.tolist()
        
    return None

def count_violations(colors, n, triples, triple_idx):
    # Count red K4s and blue K5s
    # This is O(n^4) and O(n^5) which is too slow if called repeatedly.
    # We need an incremental approach or a faster counting method.
    # For initialization, we can afford a slower count if it's not too bad.
    # n=34, n^4 is ~1.3M, n^5 is ~45M. 45M operations in Python is ~5-10s. Too slow for multiple inits.
    
    # Let's optimize count_violations using bitmasks or vectorized numpy.
    # Or just use the delta approach from a random start and assume it converges?
    # No, we need to know if we start close.
    
    # Let's implement a faster count using precomputed structures.
    # For each pair (a,b), let R_ab be the set of c such that (a,b,c) is red.
    # A red K4 {a,b,c,d} exists if c in R_ab, d in R_ab, and (c,d) is red?
    # No, (a,c,d) and (b,c,d) must also be red.
    
    # Let's stick to the delta method but initialize with a "good" guess.
    # Actually, we can compute the initial score by summing deltas from an empty state?
    # No, that's complex.
    
    # Let's try to make count_violations faster.
    # We can iterate over all 4-sets and 5-sets.
    # For n=34, 4-sets = 46376. 5-sets = 278256.
    # Checking each is O(1) if we have fast access.
    # Using a flat array and precomputed indices for the 4 and 10 triples in each set.
    
    # Precompute indices for all 4-sets and 5-sets
    # This is done once per solve_n call.
    
    # 4-sets
    sets4 = list(itertools.combinations(range(n), 4))
    # For each 4-set, store the indices of its 4 triples
    idx4 = []
    for s in sets4:
        t_idx = []
        for t in itertools.combinations(s, 3):
            t_idx.append(triple_idx[t])
        idx4.append(t_idx)
        
    # 5-sets
    sets5 = list(itertools.combinations(range(n), 5))
    idx5 = []
    for s in sets5:
        t_idx = []
        for t in itertools.combinations(s, 3):
            t_idx.append(triple_idx[t])
        idx5.append(t_idx)
        
    # Vectorized count
    # colors is a numpy array
    # For 4-sets: check if all 4 are 0 (red)
    # We can do this by summing colors at those indices. If sum == 0, it's a red K4.
    # colors[idx4[i]] -> array of 4 elements.
    # This is still a loop over 46k sets.
    # Let's try to vectorize this.
    # Create a matrix of indices? 
    # idx4_arr = np.array(idx4) # shape (num_sets4, 4)
    # red_counts4 = np.sum(colors[idx4_arr] == 0, axis=1)
    # v4 = np.sum(red_counts4 == 4)
    
    # Same for 5-sets.
    # idx5_arr = np.array(idx5) # shape (num_sets5, 10)
    # blue_counts5 = np.sum(colors[idx5_arr] == 1, axis=1)
    # v5 = np.sum(blue_counts5 == 10)
    
    idx4_arr = np.array(idx4, dtype=np.int32)
    idx5_arr = np.array(idx5, dtype=np.int32)
    
    # colors[idx4_arr] works if idx4_arr is 2D.
    # It creates a view or copy of shape (num_sets4, 4).
    # This is fast in NumPy.
    
    v4 = np.sum(np.sum(colors[idx4_arr] == 0, axis=1) == 4)
    v5 = np.sum(np.sum(colors[idx5_arr] == 1, axis=1) == 10)
    
    return int(v4), int(v5)

def calc_delta(colors, idx, n, triples, triple_idx):
    # Calculate change in score if we flip triple at idx
    a, b, c = triples[idx]
    old_color = colors[idx]
    new_color = 1 - old_color
    
    delta = 0
    
    # Check 4-sets containing (a,b,c)
    # The other vertex d can be any vertex except a,b,c
    for d in range(n):
        if d == a or d == b or d == c:
            continue
        # The 4-set is {a,b,c,d}
        # Triples in this 4-set:
        # (a,b,c) -> idx
        # (a,b,d) -> triple_idx[(min(a,b,d), ...)]
        # ...
        
        # We need to check if the 4-set was a red K4 before and after.
        # Before: (a,b,c) is old_color.
        # If old_color == 1 (blue), it's NOT a red K4.
        # If old_color == 0 (red), it MIGHT be a red K4 if the other 3 are red.
        
        # Let's get indices of the other 3 triples
        t1 = (a,b,d)
        t2 = (a,c,d)
        t3 = (b,c,d)
        
        # Sort to get canonical form
        t1 = tuple(sorted(t1))
        t2 = tuple(sorted(t2))
        t3 = tuple(sorted(t3))
        
        i1 = triple_idx[t1]
        i2 = triple_idx[t2]
        i3 = triple_idx[t3]
        
        # Check if other 3 are red (0)
        is_red_other = (colors[i1] == 0) and (colors[i2] == 0) and (colors[i3] == 0)
        
        if old_color == 0:
            # Was red. If other 3 are red, it WAS a red K4.
            if is_red_other:
                # It was a violation. After flip, it's not.
                # So v4 decreases by 1.
                delta -= 1
            else:
                # Was not a violation. After flip, still not (since it's blue now).
                pass
        else:
            # Was blue. Not a red K4.
            # After flip, it is red.
            # If other 3 are red, it BECOMES a red K4.
            if is_red_other:
                # v4 increases by 1.
                delta += 1
            else:
                pass
                
    # Check 5-sets containing (a,b,c)
    # The other 2 vertices d, e
    others = [v for v in range(n) if v not in (a,b,c)]
    for i in range(len(others)):
        for j in range(i+1, len(others)):
            d = others[i]
            e = others[j]
            
            # 5-set: {a,b,c,d,e}
            # 10 triples.
            # One is (a,b,c) -> idx
            # The other 9 must be blue (1) for it to be a blue K5.
            
            # List of other 9 triples
            verts = [a,b,c,d,e]
            all_triples = list(itertools.combinations(verts, 3))
            other_triples = [t for t in all_triples if t != tuple(sorted((a,b,c)))]
            
            # Check if all 9 are blue
            all_blue = True
            for t in other_triples:
                t_sorted = tuple(sorted(t))
                ti = triple_idx[t_sorted]
                if colors[ti] != 1:
                    all_blue = False
                    break
            
            if old_color == 1:
                # Was blue. If other 9 are blue, it WAS a blue K5.
                if all_blue:
                    # v5 decreases by 1.
                    delta -= 1
            else:
                # Was red. Not a blue K5.
                # After flip, it is blue.
                if all_blue:
                    # v5 increases by 1.
                    delta += 1
                    
    return delta
