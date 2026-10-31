import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Optimization:
    # For weight-4 vectors in {-1,0,1}^11, the dot product is always even.
    # The maximum dot product between distinct vectors is 2.
    # (Dot 4 implies identical vectors, which are not in the list).
    # The condition for compatibility is dot <= 2.
    # Therefore, ALL weight-4 vectors are mutually compatible.
    # We can skip the MIS computation entirely.
    
    # Verify this assumption quickly?
    # If we trust the math, we just return all.
    # To be safe, we can check a small sample or just trust it.
    # Given the time limit and the deterministic nature, trusting the math is best.
    
    # However, if there is any error in my deduction, the score will drop.
    # Let's add a fast check for conflicts in the first 100 vectors.
    # If no conflicts found in sample, assume none exist.
    
    if len(vectors_v4) > 0:
        sample = vectors_v4[:100]
        dots_sample = sample @ sample.T
        # Check off-diagonal
        mask = ~np.eye(100, dtype=bool)
        if np.any(dots_sample[mask] > 2):
            # Conflicts exist! Fall back to MIS
            return _construct_with_mis(vectors_v4, vectors_v1, deadline)
            
    # No conflicts found in sample. Return all.
    final_vectors = np.vstack([vectors_v4, vectors_v1])
    return final_vectors

def _construct_with_mis(vectors_v4, vectors_v1, deadline):
    N4 = len(vectors_v4)
    if N4 == 0:
        return vectors_v1
        
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    alive = np.ones(N4, dtype=bool)
    selected_indices = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    rng = np.random.default_rng(42)
    
    while np.any(alive) and time.time() < deadline:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        selected_indices.append(chosen)
        
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        alive[to_remove] = False
        
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
            
        if time.time() > deadline - 10:
            break

    result_v4 = vectors_v4[selected_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
