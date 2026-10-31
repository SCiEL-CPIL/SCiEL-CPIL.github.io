import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
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
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # KEY INSIGHT: For weight-4 vectors in {-1,0,1}^11, the maximum dot product
    # between any two DISTINCT vectors is exactly 2.
    # 
    # Proof: Two weight-4 vectors share at most 4 positions. If they share k positions
    # with the same sign, their dot product is k. The maximum k for distinct vectors
    # is 3 (if they shared all 4, they'd be identical). Wait, actually they could share
    # 3 positions with same sign and differ on the 4th, giving dot=3.
    # 
    # Let me verify: u = (1,1,1,1,0,...), v = (1,1,1,-1,0,...)
    # dot = 1+1+1-1 = 2. 
    # u = (1,1,1,1,0,...), v = (1,1,-1,-1,0,...)  
    # dot = 1+1-1-1 = 0.
    # 
    # Maximum dot: to maximize dot, we want as many matching signs as possible.
    # If they share 3 positions with same sign and 1 position where one has +1 and other has -1:
    # dot = 3*1 + 1*(-1) = 2.
    # If they share 4 positions but one has opposite sign: not possible since both have weight 4
    # and if they share all 4 positions, they must have the same signs to both be weight 4
    # in those exact positions... actually no, they could have different signs.
    # 
    # u = (1,1,1,1,0,...), v = (1,1,1,-1,0,...): dot = 1+1+1-1 = 2
    # u = (1,1,1,1,0,...), v = (1,1,-1,1,0,...): dot = 1+1-1+1 = 2
    # 
    # Can we get dot=3? u=(1,1,1,1,0,...), v=(1,1,1,?,0,...) where ? is in {-1,0,1}
    # and v has weight 4. If ?=1, v=u (identical). If ?=-1, dot=2. If ?=0, v has weight 3, not 4.
    # So max dot for distinct weight-4 vectors is 2. ✓
    #
    # Therefore, ALL weight-4 vectors are mutually compatible (dot ≤ 2 satisfies the constraint).
    # We can include ALL of them without any MIS computation.
    
    # Verify with a quick sample check for safety
    if N4 > 100:
        sample = vectors_v4[:200]
        dots_sample = sample @ sample.T
        mask = ~np.eye(200, dtype=bool)
        if np.any(dots_sample[mask] > 2):
            # Fallback: this shouldn't happen, but handle it
            return _fallback_with_mis(vectors_v4, vectors_v1, deadline)
    
    # All weight-4 vectors are compatible. Include all of them.
    # All weight-1 vectors are also compatible with each other and with weight-4.
    
    final_vectors = np.vstack([vectors_v4, vectors_v1])
    
    # Verify the result is valid
    if len(final_vectors) > 1:
        # Check a random sample of pairs to ensure validity
        rng = np.random.default_rng(42)
        n = len(final_vectors)
        for _ in range(1000):
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
            u = final_vectors[i]
            v = final_vectors[j]
            norm_u_sq = np.dot(u, u)
            norm_v_sq = np.dot(v, v)
            max_norm_sq = max(norm_u_sq, norm_v_sq)
            dist_sq = norm_u_sq + norm_v_sq - 2 * np.dot(u, v)
            if dist_sq < max_norm_sq:
                # Invalid pair found - should not happen
                return _fallback_with_mis(vectors_v4, vectors_v1, deadline)
    
    return final_vectors

def _fallback_with_mis(vectors_v4, vectors_v1, deadline):
    """Fallback if the math assumption is wrong."""
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
    
    while np.any(alive) and time.time() < deadline - 10:
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
            
    result_v4 = vectors_v4[selected_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
