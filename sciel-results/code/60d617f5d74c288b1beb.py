# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points forming a kissing configuration.
    Strategy: Use the Leech lattice structure projected to 11 dimensions, or use known
    high-density lattice constructions.
    
    The Leech lattice in 24D has 196560 minimal vectors. We can project/embed to 11D.
    However, a more direct approach: use the D_11 lattice or a combination.
    
    For dimension 11, known kissing numbers suggest we can get many points.
    Let's use a construction based on the E8 lattice product and projection.
    
    Actually, let's use a simpler but effective approach: generate points from the
    Leech lattice's minimal vectors projected to 11D, or use the fact that we can
    take the 240 minimal vectors of E8 and embed into 11D via a linear map.
    
    Better: Use the construction from the D_n lattice. D_11 has 2^10 = 1024 minimal vectors
    of norm 2 (vectors with two ±1 entries and rest 0, plus vectors with all coordinates ±1 
    with even number of minus signs... wait, let me be more careful).
    
    D_n lattice: vectors with integer coordinates summing to even, or half-integer coordinates
    summing to even. The minimal vectors of D_n are:
    - (±1, ±1, 0, ..., 0) type: n(n-1) vectors
    - (±1/2, ±1/2, ..., ±1/2) with even number of minus signs: 2^(n-1) vectors
    
    For D_11: 
    - Type 1: 11*10 = 110 vectors of norm 2
    - Type 2: 2^10 = 1024 vectors of norm 11/4 = 2.75
    
    If we scale by 2, type 1 becomes norm 4 (vectors with two ±2), type 2 becomes norm 11/2*2... 
    wait, let me just use integer vectors.
    
    Let me use a different approach: generate candidate points from a grid and use
    greedy selection to maximize the kissing configuration.
    
    Given the 600s budget, let's do a structured search:
    1. Generate candidate points from the D_11 lattice (scaled to integers)
    2. Use greedy or local search to select a maximum set satisfying the constraint
    
    The constraint is: min_pairwise_dist^2 >= max_norm^2
    
    Let's use the Leech lattice minimal vectors. The Leech lattice Λ_24 has 196560 minimal 
    vectors of norm 4. We can take a 24D to 11D projection, but that may not preserve 
    the kissing property cleanly.
    
    Alternative: Use the fact that in R^11, we can fit many unit vectors. Let's use a 
    systematic construction.
    
    For now, let's use a greedy approach with a good initial set of candidates.
    """
    start_time = time.time()
    deadline = start_time + 550  # 550s deadline, 50s safety margin
    
    d = 11
    
    # Generate candidate points from the D_11 lattice scaled to integers
    # D_11 minimal vectors (type 1): e_i ± e_j for i != j, scaled by 2 to get (±2, ±2, 0,...,0)
    # These have norm squared = 8
    # D_11 minimal vectors (type 2): (±1/2, ..., ±1/2) with even minus count, scaled by 4
    # gives (±2, ±2, ..., ±2) with even number of -2s, norm squared = 11*4 = 44
    
    # Let's use a broader set of candidates
    candidates = []
    
    # Type 1: vectors with exactly two non-zero entries, each ±1
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Type 2: vectors with all entries ±1, even number of -1s
    for bits in product([1, -1], repeat=d):
        if sum(1 for b in bits if b == -1) % 2 == 0:
            candidates.append(np.array(bits, dtype=np.int64))
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Remove duplicates and zero vector
    candidates = np.unique(candidates, axis=0)
    mask = np.any(candidates != 0, axis=1)
    candidates = candidates[mask]
    
    # Now we need to select a subset such that min_pairwise_dist^2 >= max_norm^2
    # Let's try different norm levels
    
    # First, let's try with all candidates and see what max k we can get
    # This is a maximum independent set problem in a graph where edges connect
    # points that violate the constraint.
    
    # For efficiency, let's use a greedy approach
    # Sort candidates by norm (prefer lower norm for better packing)
    norms = np.sum(candidates ** 2, axis=1)
    order = np.argsort(norms)
    candidates_sorted = candidates[order]
    norms_sorted = norms[order]
    
    def check_valid(points):
        """Check if the set of points satisfies the kissing constraint."""
        if len(points) < 2:
            return True
        n = len(points)
        max_norm = np.max(np.sum(points ** 2, axis=1))
        min_dist = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.sum(diff ** 2)
                if dist < min_dist:
                    min_dist = dist
        return min_dist >= max_norm
    
    # Greedy selection
    selected = []
    for i in range(len(candidates_sorted)):
        if time.time() > deadline:
            break
        candidate = candidates_sorted[i]
        candidate_norm = norms_sorted[i]
        
        # Try adding this candidate
        trial = selected + [candidate]
        if len(trial) < 2:
            selected = trial
            continue
        
        # Check if adding this candidate maintains validity
        # We need: min_pairwise_dist^2 >= max(max_norm_old, candidate_norm)
        trial_arr = np.array(trial)
        new_max_norm = max(np.max(np.sum(trial_arr ** 2, axis=1)), candidate_norm)
        
        # Check pairwise distances
        valid = True
        for j in range(len(selected)):
            diff = candidate - selected[j]
            dist = np.sum(diff ** 2)
            if dist < new_max_norm:
                valid = False
                break
        
        if valid:
            selected.append(candidate)
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    selected_arr = np.array(selected, dtype=np.int64) if selected else np.array([np.ones(d), -np.ones(d)], dtype=np.int64)
    
    # Verify the result
    if len(selected_arr) >= 2:
        if not check_valid(selected_arr):
            # Fall back to the trivial solution
            selected_arr = np.array([np.ones(d), -np.ones(d)], dtype=np.int64)
    
    # Ensure shape is (k, 11)
    if selected_arr.ndim == 1:
        selected_arr = selected_arr.reshape(1, -1)
    
    return selected_arr


# EVOLVE-BLOCK-END
