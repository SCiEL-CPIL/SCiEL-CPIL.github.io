import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 115.0  # 5s margin
    
    # Candidate pool: vectors with small integer coordinates
    # Support size 1, 2, 3
    # Coefficients -1, 0, 1 (maybe -2, 2 later if needed, but -1,1 is dense)
    
    candidates = []
    # Support 1
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = s
            candidates.append(v)
            
    # Support 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Support 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Initial solution: Seed (all support 2 vectors)
    # Identify indices of support 2 vectors in candidates
    # They are the ones with exactly 2 non-zeros
    norms_sq = np.sum(candidates**2, axis=1)
    non_zero_counts = np.sum(candidates != 0, axis=1)
    
    # Seed is all vectors with 2 non-zeros and norm squared 2
    seed_mask = (non_zero_counts == 2) & (norms_sq == 2)
    initial_indices = np.where(seed_mask)[0]
    
    current_set = set(initial_indices.tolist())
    
    # Helper to check validity and get metrics
    def get_metrics(indices):
        if len(indices) == 0:
            return 0, 0.0, 0.0
        vecs = candidates[list(indices)]
        norms_sq = np.sum(vecs**2, axis=1)
        max_norm = np.max(norms_sq)
        
        # Min pairwise distance squared
        # For small sets, O(n^2) is fine. For large, might be slow.
        # 220 vectors -> 220^2 = 48400 ops. Fast.
        # If we grow to 1000, 10^6 ops. Still fast in numpy.
        
        # Calculate pairwise distances
        # dist_sq[i,j] = ||v_i - v_j||^2
        # = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j
        dot = vecs @ vecs.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
        np.fill_diagonal(dist_sq, np.inf)
        
        if len(vecs) > 1:
            min_dist_sq = np.min(dist_sq)
        else:
            min_dist_sq = np.inf
            
        return len(indices), min_dist_sq, max_norm

    # Initial metrics
    k, min_d2, max_n2 = get_metrics(current_set)
    best_k = k
    best_set = set(current_set)
    
    # Local Search
    # Moves: Add, Remove, Swap
    # Accept if valid (min_d2 >= max_n2) and k increases?
    # Or accept if valid, and use simulated annealing?
    # Since we want max k, greedy add/remove might work.
    
    # Precompute candidate norms and non-zero counts for filtering?
    # We just check validity on the fly.
    
    # To speed up, we can maintain a list of "conflicting" vectors?
    # But with 220 vectors, O(n^2) check is ~50k ops.
    # 1000 iterations -> 50M ops. ~0.5s in Python? Maybe 2s.
    # 10000 iterations -> 2s.
    # We have 115s.
    
    # Let's use a simple hill climbing with random restarts or perturbations.
    
    random.seed(42)
    
    while time.time() - start_time < timeout:
        # Try to add a random vector
        if random.random() < 0.5:
            # Add
            cand_idx = random.randint(0, len(candidates) - 1)
            if cand_idx not in current_set:
                new_set = set(current_set)
                new_set.add(cand_idx)
                k_new, min_d2_new, max_n2_new = get_metrics(new_set)
                if min_d2_new >= max_n2_new:
                    current_set = new_set
                    if k_new > best_k:
                        best_k = k_new
                        best_set = new_set
        else:
            # Remove or Swap
            if len(current_set) == 0:
                continue
            if random.random() < 0.5:
                # Remove
                victim = random.choice(list(current_set))
                new_set = set(current_set)
                new_set.discard(victim)
                k_new, min_d2_new, max_n2_new = get_metrics(new_set)
                # Always valid to remove? Yes, min_d2 can only increase or stay same, max_n2 can only decrease or stay same.
                # So it's always valid.
                # But we want to maximize k. So we shouldn't remove unless we can add something better?
                # Or use it to escape local optima?
                # If we remove, k decreases. We only accept if it leads to a better state later?
                # Standard hill climbing: accept only if better.
                # But removing makes it worse.
                # So we should use "Simulated Annealing" or "Perturbation".
                
                # Let's do: Remove 1, then try to add 2?
                # Or just random walk in valid space?
                
                # Actually, if we are stuck, we might need to remove a "blocking" vector.
                # Let's accept removal with low probability (annealing).
                # Or better: Swap.
                
                # Let's implement Swap: Remove one, Add one.
                # Pick victim
                # Pick candidate to add
                # Check if new set is valid.
                
                # Re-do logic:
                pass # Placeholder
            
            # Swap Move
            victim = random.choice(list(current_set))
            cand_idx = random.randint(0, len(candidates) - 1)
            if cand_idx == victim:
                continue
            
            new_set = set(current_set)
            new_set.discard(victim)
            new_set.add(cand_idx)
            
            k_new, min_d2_new, max_n2_new = get_metrics(new_set)
            
            if min_d2_new >= max_n2_new:
                # Valid. Accept if k increased? 
                # Swap keeps k same.
                # So we accept if it allows future improvements?
                # For now, accept any valid swap to explore.
                current_set = new_set
                if k_new > best_k:
                    best_k = k_new
                    best_set = new_set
    
    # Return the best set found
    best_indices = sorted(best_set)
    if not best_indices:
        # Fallback to a single vector
        return np.array([[1] + [0]*(D-1)], dtype=np.int64)
        
    return candidates[best_indices]

# EVOLVE-BLOCK-END
