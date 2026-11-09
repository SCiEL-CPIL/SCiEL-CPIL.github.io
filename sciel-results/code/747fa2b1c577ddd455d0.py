import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    D = 11
    
    # 1. Generate the 224 vectors of type (+/- 1, +/- 1, 0, ..., 0)
    base_vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    base_vectors.append(v)
    
    # Convert to numpy array for faster distance checks
    base_arr = np.array(base_vectors, dtype=np.int64)
    base_set = set([tuple(v) for v in base_vectors])
    
    selected = list(base_vectors)
    
    # 2. Generate candidates of type (+/- 1, +/- 1, +/- 1, +/- 1, 0, ..., 0)
    # These have norm squared 4.
    # Condition: dist^2 >= 4? No.
    # If we mix norms, the condition is dist(v, w) >= max(norm(v), norm(w)).
    # Base vectors have norm sqrt(2). Candidates have norm sqrt(4)=2.
    # So dist(base, cand) >= 2 => dist^2 >= 4.
    # dist(cand, cand) >= 2 => dist^2 >= 4.
    
    # Wait, if I add vectors of norm 2, the min distance for the whole set must be >= 2.
    # The base vectors have min distance sqrt(2).
    # If I add a vector of norm 2, the distance to base vectors must be >= 2.
    # The distance between base vectors is sqrt(2) < 2.
    # So I CANNOT mix vectors of norm sqrt(2) and norm 2 if the condition is "min pairwise distance >= max vector norm".
    # Let's re-read the task carefully.
    # "minimum pairwise distance is at least the maximum vector norm"
    # Let R_max = max(||v_i||).
    # For all i != j, ||v_i - v_j|| >= R_max.
    # If I have vectors of norm sqrt(2) and vectors of norm 2, R_max = 2.
    # Then distance between two norm-sqrt(2) vectors must be >= 2.
    # But in the base set, distance is sqrt(2).
    # So I cannot simply add norm-2 vectors to a set of norm-sqrt(2) vectors unless I remove some base vectors or change the base vectors.
    
    # Therefore, the 252 vectors must ALL have the same norm, or the "max norm" is determined by the largest norm present, and ALL pairwise distances must meet that.
    # If the 252 vectors in 11D are all norm sqrt(2), then min dist must be sqrt(2).
    # Are the 252 vectors all norm sqrt(2)?
    # The 224 are norm sqrt(2).
    # If the other 28 are also norm sqrt(2), they must be of the form +/- e_i +/- e_j.
    # But we already used all of those.
    # So the other 28 MUST be of a different form, or my assumption that the 224 are all in the set is wrong.
    
    # Alternative: The 252 vectors are all of norm sqrt(2)?
    # Are there integer vectors of norm sqrt(2) other than +/- e_i +/- e_j?
    # No, sum of squares = 2 implies two 1s (or -1s) and rest 0.
    # So there are ONLY 224 such vectors in R^11.
    # So we CANNOT have 252 vectors all of norm sqrt(2).
    
    # Therefore, the 252 vectors must have a larger max norm.
    # If max norm is sqrt(3), then min dist must be sqrt(3).
    # Can we have 252 vectors of norm sqrt(3) with min dist sqrt(3)?
    # Vectors of norm sqrt(3) are of type (+/- 1, +/- 1, +/- 1, 0...).
    # Count: C(11, 3) * 2^3 = 165 * 8 = 1320.
    # We need to pick 252 of these such that pairwise dist^2 >= 3.
    
    # Or, the configuration is a mix, but the max norm is larger.
    # Actually, the standard 252 configuration in 11D is often cited as having vectors of norm sqrt(2) and sqrt(3)?
    # No, usually kissing configurations are "equi-norm" after normalization.
    # The task says "Normalizing each vector to norm 2 certifies...".
    # This implies the raw vectors can have different norms, but the DISTANCE condition is based on the RAW norms.
    # "minimum pairwise distance is at least the maximum vector norm"
    
    # If I use the 224 vectors of norm sqrt(2), R_max = sqrt(2).
    # Min dist is sqrt(2). This is valid. Score 224.
    
    # To get 252, I need to either:
    # 1. Find 28 more vectors of norm sqrt(2) (Impossible, only 224 exist).
    # 2. Replace some or all with vectors of higher norm, such that the new R_max allows 252 vectors.
    # If I use vectors of norm sqrt(3), R_max = sqrt(3).
    # I need 252 vectors of norm sqrt(3) with pairwise dist >= sqrt(3).
    
    # Let's try to find 252 vectors of norm sqrt(3) (type 1,1,1,0...).
    # Greedy search over the 1320 candidates.
    
    candidates = []
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = [0]*D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
                            
    # Shuffle for greedy
    import random
    random.shuffle(candidates)
    
    selected = []
    selected_set = set()
    
    for v in candidates:
        v_tuple = tuple(v)
        if v_tuple in selected_set:
            continue
            
        if not selected:
            selected.append(v)
            selected_set.add(v_tuple)
            continue
            
        # Check distance against all selected
        # v has norm sqrt(3). Selected have norm sqrt(3).
        # Need dist^2 >= 3.
        
        ok = True
        for u in selected:
            diff = np.array(v) - np.array(u)
            if np.dot(diff, diff) < 3:
                ok = False
                break
                
        if ok:
            selected.append(v)
            selected_set.add(v_tuple)
            
        if len(selected) >= 252:
            break
            
    if len(selected) > 224:
        return np.array(selected, dtype=np.int64)
    else:
        return np.array(base_vectors, dtype=np.int64)
