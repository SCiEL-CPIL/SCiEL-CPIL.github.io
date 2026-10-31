"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate all weight 2 vectors
    vecs_w2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vecs_w2.append(v)
    
    # Generate all weight 3 vectors
    vecs_w3 = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    vecs_w3.append(v)
    
    # Convert to numpy for faster distance calculations if needed, 
    # but list of lists is fine for small D and moderate N.
    # We will use a greedy approach.
    
    best_set = [np.array(v) for v in vecs_w2] # Start with all W2 (known valid)
    best_score = len(best_set)
    
    # Try to improve by mixing W2 and W3
    # We want to find a subset of W2 and W3 such that min dist sq >= 3
    # (since max norm sq is 3 from W3)
    
    # If we only use W2, min dist sq is 2. Max norm sq is 2. Valid.
    # If we use W3, max norm sq is 3. We need min dist sq >= 3.
    
    # Greedy: Shuffle all candidates, add if compatible with current set.
    # We need to check compatibility with ALL existing vectors.
    
    candidates = vecs_w2 + vecs_w3
    np.random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        np.random.shuffle(candidates)
        current_set = []
        current_sq_norms = []
        
        for v in candidates:
            v_arr = np.array(v)
            norm_sq = np.dot(v_arr, v_arr)
            
            # Check compatibility
            is_compatible = True
            for i, existing in enumerate(current_set):
                dist_sq = np.sum((v_arr - existing) ** 2)
                # Condition: dist_sq >= max(norm_sq, existing_norm_sq)
                # Actually, condition is dist_sq >= max_norm_sq_in_set
                # But max_norm_sq_in_set is determined by the set.
                # If we add a W3 vector, max_norm_sq becomes 3.
                # So if current_set has any W3, we need dist_sq >= 3.
                # If current_set has only W2, we need dist_sq >= 2.
                
                # To simplify, let's enforce the stricter condition: dist_sq >= 3.
                # This ensures validity for both W2-only and Mixed sets.
                # If dist_sq >= 3, it is also >= 2.
                # So we can just check dist_sq >= 3.
                
                if dist_sq < 3:
                    is_compatible = False
                    break
            
            if is_compatible:
                current_set.append(v_arr)
                current_sq_norms.append(norm_sq)
        
        if len(current_set) > best_score:
            best_score = len(current_set)
            best_set = current_set
            # If we hit 630 or high, we can stop early?
            if best_score > 400:
                break

    return np.array(best_set, dtype=np.int64)
# EVOLVE-BLOCK-END
