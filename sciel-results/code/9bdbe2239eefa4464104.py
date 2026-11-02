"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def _generate_candidates(weights):
    """Generate all integer vectors with the specified weights."""
    candidates = []
    for w in weights:
        if w > D:
            continue
        for indices in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Generate candidates with weight 3 and 4
    candidates = _generate_candidates([3, 4])
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # We'll use a hybrid approach:
    # 1. Greedy with multiple restarts to find a large initial set
    # 2. Local search (simulated annealing) to try to improve it
    
    best_vectors = []
    best_count = 0
    
    max_restarts = 200
    rng = np.random.default_rng(42)
    
    # Greedy phase
    for restart in range(max_restarts):
        if time.time() > deadline - 30:
            break
            
        perm = rng.permutation(n_cand)
        selected_indices = []
        selected_vectors = []
        
        for idx in perm:
            v = candidates[idx]
            n_v = norms_sq[idx]
            
            if not selected_vectors:
                selected_indices.append(idx)
                selected_vectors.append(v)
                continue
            
            sel_arr = np.array(selected_vectors)
            sel_norms = np.array([norms_sq[i] for i in selected_indices])
            
            dots = sel_arr @ v
            dist_sq = n_v + sel_norms - 2 * dots
            max_norms = np.maximum(n_v, sel_norms)
            
            if np.all(dist_sq >= max_norms):
                selected_indices.append(idx)
                selected_vectors.append(v)
        
        if len(selected_vectors) > best_count:
            best_count = len(selected_vectors)
            best_vectors = selected_vectors.copy()
    
    # Local search / simulated annealing phase
    if best_vectors and time.time() < deadline - 10:
        # Build a conflict graph for the current best set
        # And try to improve via swaps
        
        current_set = np.array(best_vectors)
        current_norms = np.sum(current_set * current_set, axis=1)
        current_count = len(current_set)
        
        # Build set of selected indices (approximate)
        # For local search, we'll work with the vectors directly
        
        # Try to add vectors from the candidate pool that aren't in the current set
        # Build a set of current vectors for fast lookup
        current_vec_set = set(map(tuple, current_set))
        
        # Find candidates not in current set
        available_candidates = []
        for i in range(n_cand):
            if tuple(candidates[i]) not in current_vec_set:
                available_candidates.append(i)
        
        available_candidates = np.array(available_candidates)
        
        # Simulated annealing
        if len(available_candidates) > 0:
            temp = 10.0
            cooling = 0.95
            min_temp = 0.1
            rng2 = np.random.default_rng(123)
            
            for step in range(50000):
                if time.time() > deadline - 5:
                    break
                if temp < min_temp:
                    break
                
                # Pick a random available candidate
                idx = available_candidates[rng2.integers(0, len(available_candidates))]
                v = candidates[idx]
                n_v = norms_sq[idx]
                
                # Check if v is compatible with current set
                dots = current_set @ v
                dist_sq = n_v + current_norms - 2 * dots
                max_norms = np.maximum(n_v, current_norms)
                compatible = np.all(dist_sq >= max_norms)
                
                if compatible:
                    # Add it
                    current_set = np.vstack([current_set, v])
                    current_norms = np.append(current_norms, n_v)
                    current_count += 1
                    current_vec_set.add(tuple(v))
                    # Remove from available
                    mask = ~np.array([tuple(candidates[i]) == tuple(v) for i in available_candidates])
                    available_candidates = available_candidates[mask]
                else:
                    # Try a swap: remove a random vector from current set, add v
                    if current_count > 1:
                        remove_idx = rng2.integers(0, current_count)
                        # Check if removing remove_idx and adding v is valid
                        # This is complex, so we'll just do simple addition
                        pass
                
                temp *= cooling
        
        # Final greedy pass to add any remaining compatible vectors
        current_vec_set = set(map(tuple, current_set))
        for i in range(n_cand):
            if time.time() > deadline - 2:
                break
            if tuple(candidates[i]) not in current_vec_set:
                v = candidates[i]
                n_v = norms_sq[i]
                dots = current_set @ v
                dist_sq = n_v + current_norms - 2 * dots
                max_norms = np.maximum(n_v, current_norms)
                if np.all(dist_sq >= max_norms):
                    current_set = np.vstack([current_set, v])
                    current_norms = np.append(current_norms, n_v)
                    current_vec_set.add(tuple(v))
        
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_vectors = current_set.copy()
    
    if not best_vectors:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
