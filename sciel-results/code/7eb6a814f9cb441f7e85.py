import numpy as np
import itertools
import time
import random

D = 11

def _gen_weight4():
    """Generate all weight-4 vectors in {-1,0,1}^11 (norm^2=4)."""
    out = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            out.append(v)
    return np.array(out, dtype=np.int64)

def _gen_weight1():
    """Generate all weight-1 vectors in {-2,0,2}^11 (norm^2=4)."""
    out = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            out.append(v)
    return np.array(out, dtype=np.int64)

def _gen_weight2():
    """Generate all weight-2 vectors with norm^2=4: two entries of +-2 or one entry of +-2 and... wait.
    Weight-2 vectors in {-1,0,1}^11 have norm^2=2, not 4.
    For norm^2=4 with weight 2: entries must be +-2 (norm^2=4+0=4) - that's weight 1.
    Or one entry +-2 and another... no, weight means number of non-zero entries.
    Actually for integer vectors with norm^2=4:
    - weight 1: one entry +-2 (norm^2=4)
    - weight 2: two entries, each +-... 2^2=4, so can't have two nonzeros summing to 4 unless 0^2+2^2=4 (weight 1) or 1^2+1^2=2 (weight 2, norm^2=2)
    - weight 4: four entries +-1 (norm^2=4)
    
    So weight-2 with norm^2=4 doesn't exist in integer vectors.
    Let me reconsider: maybe we should include weight-2 vectors with norm^2=2? No, the constraint is min pairwise dist >= max norm.
    
    Actually, let's stick with norm^2=4 vectors only for now.
    """
    return np.array([], dtype=np.int64).reshape(0, D)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Generate candidate pools - all norm^2=4 vectors
    v4_candidates = _gen_weight4()
    v1_candidates = _gen_weight1()
    
    # Combine all norm^2=4 candidates
    all_candidates = np.vstack([v4_candidates, v1_candidates])
    N = len(all_candidates)
    
    if N == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # Compute all pairwise dot products
    dots = all_candidates @ all_candidates.T
    
    # For vectors with norm^2=4: dist^2 = 4 + 4 - 2*u.v = 8 - 2*u.v
    # Need dist^2 >= 4 (max norm^2), so 8 - 2*u.v >= 4 => u.v <= 2
    # Conflict if u.v > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Build adjacency lists and degrees
    adj = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    # Farthest-point sampling style: iteratively add vector that maximizes min distance to current set
    # This is equivalent to finding a maximal independent set with a greedy distance-based heuristic
    
    # First, let's use a better initial set: pick vectors that are "spread out"
    # Start with a random seed vector
    rng = random.Random(42)
    
    # Compute minimum squared distance from each candidate to a growing set
    # We'll use a greedy approach: at each step, add the candidate that maximizes the minimum
    # squared distance to all already-selected candidates
    
    selected = []
    selected_set = set()
    
    # Initialize with a random candidate
    first_choice = rng.randrange(N)
    selected.append(first_choice)
    selected_set.add(first_choice)
    
    # min_dist_sq[i] = minimum squared distance from candidate i to any selected candidate
    # dist_sq[i][j] = ||v_i - v_j||^2 = 8 - 2*dots[i][j] for norm^2=4 vectors
    dist_sq = 8 - 2 * dots
    np.fill_diagonal(dist_sq, 0)
    
    # Initialize min_dist_sq
    min_dist_sq = dist_sq[first_choice].copy()
    
    # Iteratively add the candidate with maximum min_dist_sq that is compatible
    # (i.e., min_dist_sq >= 4, which means no conflicts)
    available = np.ones(N, dtype=bool)
    available[first_choice] = False
    
    while time.time() < deadline - 10:
        # Find the candidate with maximum min_dist_sq among available candidates
        # that is compatible (min_dist_sq >= 4)
        candidate_indices = np.where(available & (min_dist_sq >= 4))[0]
        
        if len(candidate_indices) == 0:
            break
        
        # Pick the one with maximum min_dist_sq (ties broken randomly)
        max_min_dist = np.max(min_dist_sq[candidate_indices])
        best_candidates = candidate_indices[min_dist_sq[candidate_indices] == max_min_dist]
        chosen = best_candidates[rng.randrange(len(best_candidates))]
        
        selected.append(chosen)
        selected_set.add(chosen)
        available[chosen] = False
        
        # Update min_dist_sq for remaining available candidates
        # min_dist_sq[i] = min(min_dist_sq[i], dist_sq[i][chosen])
        min_dist_sq = np.minimum(min_dist_sq, dist_sq[chosen])
        
        # Remove candidates that now conflict (min_dist_sq < 4)
        available &= (min_dist_sq >= 4)
    
    best_set = set(selected)
    best_size = len(best_set)
    
    # Local improvement: try to add vertices that don't conflict with current set
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for v in range(N):
            if v in best_set:
                continue
            # Check if v can be added (no conflicts with current set)
            if np.any(conflicts[v][list(best_set)]):
                continue
            best_set.add(v)
            best_size += 1
            improved = True
    
    # Try random restarts if time permits
    if time.time() < deadline - 8:
        for restart in range(5):
            if time.time() >= deadline - 6:
                break
            
            rng_r = random.Random(42 + restart * 1000)
            
            # Farthest-point sampling restart
            selected_r = []
            first_choice_r = rng_r.randrange(N)
            selected_r.append(first_choice_r)
            selected_set_r = {first_choice_r}
            
            min_dist_sq_r = dist_sq[first_choice_r].copy()
            available_r = np.ones(N, dtype=bool)
            available_r[first_choice_r] = False
            
            while time.time() < deadline - 6:
                candidate_indices_r = np.where(available_r & (min_dist_sq_r >= 4))[0]
                if len(candidate_indices_r) == 0:
                    break
                max_min_dist_r = np.max(min_dist_sq_r[candidate_indices_r])
                best_candidates_r = candidate_indices_r[min_dist_sq_r[candidate_indices_r] == max_min_dist_r]
                chosen_r = best_candidates_r[rng_r.randrange(len(best_candidates_r))]
                selected_r.append(chosen_r)
                selected_set_r.add(chosen_r)
                available_r[chosen_r] = False
                min_dist_sq_r = np.minimum(min_dist_sq_r, dist_sq[chosen_r])
                available_r &= (min_dist_sq_r >= 4)
            
            set_r = set(selected_r)
            
            # Local improvement
            imp_r = True
            while imp_r and time.time() < deadline - 6:
                imp_r = False
                for v in range(N):
                    if v in set_r:
                        continue
                    if np.any(conflicts[v][list(set_r)]):
                        continue
                    set_r.add(v)
                    imp_r = True
            
            if len(set_r) > best_size:
                best_set = set_r
                best_size = len(set_r)
    
    result_vectors = all_candidates[np.array(list(best_set))]
    
    return result_vectors
