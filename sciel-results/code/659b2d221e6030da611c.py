import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # 1. Generate all weight-6 vectors in R^11
    # A weight-6 vector has exactly 6 non-zero entries, each +/- 1.
    # Indices: choose 6 from 11. Signs: 2^6.
    candidates = []
    indices_iter = itertools.combinations(range(D), 6)
    for idx in indices_iter:
        # Generate all sign combinations for these 6 indices
        # Use product for signs
        for signs in itertools.product([-1, 1], repeat=6):
            v = [0] * D
            for i, s in zip(idx, signs):
                v[i] = s
            candidates.append(tuple(v))
    
    # Candidates are tuples of ints
    # Constraint for weight-6 set:
    # Max norm squared = 6.
    # Min dist squared >= 6.
    # ||v - u||^2 = 6 + 6 - 2 v.u = 12 - 2 v.u >= 6
    # 2 v.u <= 6 => v.u <= 3
    # So we need an independent set in the graph where edges connect v, u if v.u > 3.
    
    n_cand = len(candidates)
    best_set = []
    best_score = 0
    
    # Pre-compute dot products? No, too much memory (29k^2).
    # We will compute on the fly.
    
    # Helper to check if a vector can be added to a set
    def can_add(v, current_set):
        for u in current_set:
            # Compute dot product
            dot = 0
            for i in range(D):
                dot += v[i] * u[i]
            if dot > 3:
                return False
        return True

    # Greedy approach with random shuffling
    num_attempts = 20
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
            
        shuffled_cands = candidates.copy()
        random.shuffle(shuffled_cands)
        
        current_set = []
        
        for v in shuffled_cands:
            if time.time() > deadline:
                break
            
            # Check against current_set
            ok = True
            for u in current_set:
                dot = 0
                for i in range(D):
                    dot += v[i] * u[i]
                if dot > 3:
                    ok = False
                    break
            
            if ok:
                current_set.append(v)
        
        if len(current_set) > best_score:
            best_score = len(current_set)
            best_set = current_set.copy()
            
        # Optional: Local Search / Polish
        # Try to add any remaining vector
        # This is expensive if done naively.
        # Simple polish: try to swap a member of current_set with a non-member
        # to allow adding more?
        # Given time, simple greedy with multiple restarts is often strong for this size.
        
    # Convert best_set to numpy array
    if not best_set:
        # Fallback to weight-2 if something went wrong
        # Weight 2 always works with score 220
        w2_cands = []
        for idx in itertools.combinations(range(D), 2):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    v = [0] * D
                    v[idx[0]] = s1
                    v[idx[1]] = s2
                    w2_cands.append(v)
        return np.array(w2_cands, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)
