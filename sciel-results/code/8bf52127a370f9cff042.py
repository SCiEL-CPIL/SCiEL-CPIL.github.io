import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Generate candidate vectors with small norms
    # We'll try vectors with entries in {-1, 0, 1} first, maybe {-2, -1, 0, 1, 2}
    # Start with the seed: size 2, +/-1
    candidates = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # Add size 3 vectors
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
    
    # Add size 1 vectors
    for i in range(D):
        for si in (1, -1):
            v = [0] * D
            v[i] = si
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(candidates**2, axis=1)
    
    # We want to select a subset S such that min_dist_sq >= max_norm_sq
    # Greedy approach:
    # 1. Sort candidates by norm (ascending? descending?)
    #    If we pick a high norm vector, we restrict others.
    #    If we pick low norm vectors, we can pack more?
    #    But max norm is determined by the max in S.
    #    Strategy: Try to maximize count.
    #    Maybe fix max_norm_sq to a value N, and find max independent set in graph where edge if dist_sq < N.
    #    Then try different N.
    
    best_set = []
    best_score = 0
    
    # Possible max_norm_sq values from candidates
    unique_norms = np.unique(norms_sq)
    
    for target_norm_sq in unique_norms:
        if time.time() > deadline:
            break
            
        # Filter candidates with norm_sq <= target_norm_sq
        # Actually, if we include a vector with norm_sq > target, max_norm increases.
        # So we should only consider candidates with norm_sq == target_norm_sq as the "max" ones?
        # Or just ensure all selected have norm_sq <= target_norm_sq.
        # And min_dist_sq >= target_norm_sq.
        
        mask = norms_sq <= target_norm_sq
        sub_candidates = candidates[mask]
        sub_norms_sq = norms_sq[mask]
        
        if len(sub_candidates) == 0:
            continue
            
        # Build conflict graph for these candidates
        # Edge if dist_sq < target_norm_sq
        # This is O(N^2 * D). N can be large.
        # N for norm_sq <= 3:
        # Size 1: 22
        # Size 2: 220
        # Size 3: 1320
        # Total ~ 1562.
        # 1562^2 ~ 2.4M pairs. 2.4M * 11 ~ 26M ops. Fast in numpy?
        
        # We want max independent set.
        # Greedy: Sort by degree? Or random order?
        # Let's try a simple greedy: pick vector with fewest conflicts?
        # Or just random shuffle and greedy pick.
        
        # Calculate all pair distances
        # dist_sq[i, j] = ||u_i - u_j||^2
        # Use broadcasting
        # sub_candidates shape (N, D)
        # diff = sub_candidates[:, np.newaxis, :] - sub_candidates[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=2)
        
        N = len(sub_candidates)
        if N > 5000: # Too large for full matrix?
            # Sample or skip
            continue
            
        diff = sub_candidates[:, np.newaxis, :] - sub_candidates[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Conflict if dist_sq < target_norm_sq
        # Diagonal is 0, which is < target. We ignore self.
        conflict_matrix = (dist_sq < target_norm_sq)
        np.fill_diagonal(conflict_matrix, False)
        
        # Greedy Independent Set
        # Randomize order
        order = np.random.permutation(N)
        selected = np.zeros(N, dtype=bool)
        
        # Optimization: Maintain list of available candidates
        # Simple loop
        for i in order:
            if selected[i]:
                continue
            # Check if i conflicts with any already selected
            # selected conflicts with i: conflict_matrix[i, selected]
            # If any True, skip
            if not np.any(conflict_matrix[i, selected]):
                selected[i] = True
                
        count = np.sum(selected)
        if count > best_score:
            best_score = count
            best_set = sub_candidates[selected]
            
    if not best_set:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_set = np.array(vectors, dtype=np.int64)
        
    return best_set
