import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Generate seed vectors (weight 2, +/- 1)
    seed_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                seed_vectors.append(v)
    
    selected = np.array(seed_vectors, dtype=np.int64)
    if len(selected) == 0:
        return selected
        
    # Precompute norms squared for selected
    selected_norms_sq = np.sum(selected**2, axis=1)
    
    # 2. Generate candidate pool (components -1, 0, 1)
    # Exclude the seed vectors to avoid duplicates
    # We can generate all and filter, or generate all and check.
    # 3^11 = 177147 vectors.
    # Generating them all might take a bit of time/memory, but it's fine.
    
    # Efficient generation
    # We can use np.meshgrid or recursive generation.
    # Let's use a simple loop or itertools.product.
    candidates = []
    # To save time, we can just iterate through product
    # But 177k iterations in Python is slow.
    # Let's generate them in a more vectorized way if possible, or just accept the loop time.
    # 177k iterations is ~0.1-0.5s.
    
    # We will skip the zero vector and the seed vectors.
    # We can identify seed vectors by norm sq == 2 and weight == 2.
    
    # Let's just generate all non-zero vectors in {-1,0,1}^D
    # and filter out those already in selected.
    
    # To speed up filtering, we can use a set of tuples.
    selected_set = set(map(tuple, selected))
    
    # Generate candidates
    # Using itertools.product is standard.
    for combo in itertools.product([-1, 0, 1], repeat=D):
        if sum(x*x for x in combo) == 0:
            continue
        v_tuple = combo
        if v_tuple in selected_set:
            continue
        candidates.append(v_tuple)
        
    # Shuffle candidates
    np.random.shuffle(candidates)
    
    # Greedy insertion
    for v_tuple in candidates:
        if time.time() > deadline:
            break
            
        v = np.array(v_tuple, dtype=np.int64)
        v_norm_sq = np.sum(v**2)
        
        # Check against selected
        # dists = ||u - v||^2
        # diff = selected - v
        # dists = np.sum(diff**2, axis=1)
        # max_norms = np.maximum(selected_norms_sq, v_norm_sq)
        # valid = np.all(dists >= max_norms)
        
        # Optimization: 
        # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v
        # Condition: ||u||^2 + ||v||^2 - 2 u.v >= max(||u||^2, ||v||^2)
        # Case 1: ||u||^2 >= ||v||^2
        #   ||u||^2 + ||v||^2 - 2 u.v >= ||u||^2
        #   ||v||^2 - 2 u.v >= 0
        #   2 u.v <= ||v||^2
        # Case 2: ||v||^2 > ||u||^2
        #   ||u||^2 + ||v||^2 - 2 u.v >= ||v||^2
        #   ||u||^2 - 2 u.v >= 0
        #   2 u.v <= ||u||^2
        
        # So condition is: 2 u.v <= min(||u||^2, ||v||^2) is WRONG.
        # It's:
        # If ||u||^2 >= ||v||^2, need 2 u.v <= ||v||^2
        # If ||v||^2 > ||u||^2, need 2 u.v <= ||u||^2
        # Combined: 2 u.v <= min(||u||^2, ||v||^2) is NOT correct.
        # Let's re-evaluate.
        # If ||u||^2 >= ||v||^2, max is ||u||^2.
        # ||u-v||^2 >= ||u||^2 => 2||u||^2 - 2u.v >= ||u||^2 => ||u||^2 - 2u.v >= 0 => 2u.v <= ||u||^2.
        # Wait, my previous derivation was:
        # ||u-v||^2 = ||u||^2 + ||v||^2 - 2u.v
        # If max is ||u||^2:
        # ||u||^2 + ||v||^2 - 2u.v >= ||u||^2 => ||v||^2 - 2u.v >= 0 => 2u.v <= ||v||^2.
        # If max is ||v||^2:
        # ||u||^2 + ||v||^2 - 2u.v >= ||v||^2 => ||u||^2 - 2u.v >= 0 => 2u.v <= ||u||^2.
        
        # So:
        # If ||u||^2 >= ||v||^2, we need 2u.v <= ||v||^2.
        # If ||v||^2 > ||u||^2, we need 2u.v <= ||u||^2.
        
        # This can be written as:
        # 2u.v <= min(||u||^2, ||v||^2) is FALSE.
        # If ||u||^2 >= ||v||^2, min is ||v||^2. Condition: 2u.v <= ||v||^2. Correct.
        # If ||v||^2 > ||u||^2, min is ||u||^2. Condition: 2u.v <= ||u||^2. Correct.
        # Yes! The condition is 2 u.v <= min(||u||^2, ||v||^2).
        
        # This is much faster to compute!
        # We need to check: 2 * (selected @ v) <= np.minimum(selected_norms_sq, v_norm_sq)
        
        dots = selected @ v
        min_norms = np.minimum(selected_norms_sq, v_norm_sq)
        valid = np.all(2 * dots <= min_norms)
        
        if valid:
            selected = np.vstack([selected, v])
            selected_norms_sq = np.append(selected_norms_sq, v_norm_sq)
            selected_set.add(v_tuple)
            
    return selected
# EVOLVE-BLOCK-END
