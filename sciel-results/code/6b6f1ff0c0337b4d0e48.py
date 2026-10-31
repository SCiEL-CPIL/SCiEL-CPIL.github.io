import numpy as np
import itertools

def construct():
    D = 11
    
    # Generate all weight-4 vectors: four ±1's, rest 0 (norm^2 = 4)
    # These are the minimal vectors of the relevant lattice structure
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = 4
    
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    # For weight-4 vectors, two vectors conflict if their distance squared < 4
    # dist_sq = 8 - 2*dot, so conflict when 8 - 2*dot < 4 => dot > 2
    # For weight-4 vectors, dot can be -4,-3,-2,-1,0,1,2,3,4
    # dot > 2 means dot >= 3, which means they share at least 3 coordinates with same sign
    
    best_selected = []
    best_size = 0
    
    np.random.seed(42)
    
    # Use greedy algorithm with randomization
    # For each attempt, randomly order candidates and greedily select non-conflicting ones
    for attempt in range(50):
        order = np.random.permutation(n)
        selected = []
        selected_set = set()
        
        for idx in order:
            # Check if idx conflicts with any already selected
            if not np.any(conflict[idx, selected]):
                selected.append(idx)
                selected_set.add(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
    
    result = candidates[best_selected]
    return result
