"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product, combinations

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy: Use the D11 root lattice (octonionic/E8-based construction) which
    is known to have a kissing number of 264 in dimension 11.
    
    The D11 lattice consists of integer vectors with even sum of coordinates.
    The shortest vectors have norm^2 = 2 (vectors with two +/-1 entries and rest 0).
    There are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    
    But we can also include vectors of norm^2 = 4 that maintain the distance property.
    
    Actually, let me reconsider. The kissing number of D11 is 220 (the short vectors).
    But the Leech lattice projected to 11D might give more.
    
    Let me use a different approach: construct vectors from the D11 lattice and verify
    the condition. The condition is: min pairwise distance >= max norm.
    
    For vectors v_i, v_j: distance = ||v_i - v_j||. We need ||v_i - v_j|| >= max_i ||v_i||.
    
    If all vectors have the same norm r, then we need ||v_i - v_j|| >= r for all i != j.
    ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i·v_j = 2r^2 - 2 v_i·v_j >= r^2
    => 2r^2 - 2 v_i·v_j >= r^2
    => r^2 >= 2 v_i·v_j
    => v_i·v_j <= r^2/2
    
    For D11 short vectors (norm^2 = 2, r^2 = 2): v_i·v_j <= 1.
    Two distinct short vectors in D11: their dot product is in {-1, 0, 1}.
    - If they share one coordinate with same sign: dot = 1. 1 <= 1, OK.
    - If they share one coordinate with opposite sign: dot = -1. OK.
    - If they share no coordinates: dot = 0. OK.
    - If they share both coordinates: they're the same vector (excluded).
    
    So all 220 short vectors of D11 satisfy the condition with r^2 = 2.
    
    Can we add more? Vectors of norm^2 = 4 in D11. If we mix norms, the condition becomes
    more complex. Let's stick with equal norm for simplicity and maximize count.
    
    Actually, let me try a greedy approach to add more vectors.
    """
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Generate all short vectors of D11: integer vectors with even sum, norm^2 = 2
    # These have exactly two nonzero entries, each +/-1, and the rest 0.
    # Sum of coordinates = +/-1 +/-1 which can be -2, 0, or 2. Even sum required.
    # Sum = 0 when signs are opposite, sum = +/-2 when same sign. Both are even.
    # So all such vectors are in D11.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    
    # Verify the condition
    def check_valid(vecs):
        if len(vecs) == 0:
            return True
        norms_sq = np.sum(vecs * vecs, axis=1)
        max_norm = np.max(norms_sq)
        # Compute all pairwise distances
        # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i·v_j
        # We need min_{i!=j} ||v_i - v_j|| >= sqrt(max_norm)
        # i.e., min_{i!=j} ||v_i - v_j||^2 >= max_norm
        n = len(vecs)
        if n < 2:
            return True
        # Use matrix computation
        dot = vecs @ vecs.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        return min_dist_sq >= max_norm
    
    if not check_valid(vectors):
        # Shouldn't happen, but fallback
        return vectors
    
    # Try to add more vectors greedily
    # Consider vectors of norm^2 = 4 in D11 (even sum of coordinates)
    # Types of norm^2=4 integer vectors:
    # (a) One entry is +/-2, rest 0. Sum = +/-2, even. OK.
    # (b) Two entries are +/-1, +/-1, rest 0. Sum can be -2,0,2. Even. OK.
    # (c) Four entries are +/-1. Sum can be -4,-2,0,2,4. Even. OK.
    
    # Type (a): 11 * 2 = 22 vectors with a single +/-2
    # Type (b): already included (these are the short vectors, norm^2=2... wait no)
    # Wait, type (b) with two +/-1 entries has norm^2 = 1+1 = 2, not 4.
    # So for norm^2=4:
    # (a) Single +/-2: norm^2 = 4. 22 vectors.
    # (b) Two entries of +/-2? No, that's norm^2=8.
    # (c) One entry +/-2 and... no, that's already (a) if rest are 0.
    # Actually for norm^2=4 with integer entries:
    # - One coordinate is +/-2, rest 0: 22 vectors
    # - Two coordinates are each +/-1 and +/-1? No, 1+1=2 not 4.
    # - Four coordinates are each +/-1: 1+1+1+1=4. C(11,4)*16 = 330*16 = 5280 vectors.
    # - One coordinate +/-2 and one coordinate... no, 4+1=5.
    # - Two coordinates: one is +/-2, other is 0. That's case (a).
    
    # So norm^2=4 vectors in D11:
    # Type A: single +/-2. 22 vectors. Sum = +/-2, even. All in D11.
    # Type B: four +/-1 entries. C(11,4)*16 = 5280 vectors. Sum is even (sum of four +/-1 is even). All in D11.
    
    # Let's try adding these greedily.
    
    # First, let's try type A vectors (single +/-2)
    candidate_vectors = []
    for i in range(D):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            candidate_vectors.append(v)
    
    # Then type B vectors (four +/-1 entries)
    for idxs in combinations(range(D), 4):
        for signs in product((1, -1), repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidate_vectors.append(v)
    
    candidate_vectors = np.array(candidate_vectors, dtype=np.int64)
    
    # Greedy addition
    current = vectors.copy()
    norms_sq_current = np.sum(current * current, axis=1)
    max_norm_current = np.max(norms_sq_current)
    
    for cv in candidate_vectors:
        if time.time() - start_time > deadline - 5:
            break
        
        cv_norm_sq = int(np.sum(cv * cv))
        # New max norm would be max(max_norm_current, cv_norm_sq)
        new_max_norm = max(max_norm_current, cv_norm_sq)
        
        # Check if adding cv maintains the condition
        # Need: for all existing v_i, ||cv - v_i||^2 >= new_max_norm
        # And: ||cv - v_j||^2 >= new_max_norm for all existing pairs (unchanged)
        # The existing pairs are already valid for max_norm_current.
        # If new_max_norm > max_norm_current, we need to recheck existing pairs.
        
        # For simplicity, only add if cv_norm_sq == max_norm_current (equal norm)
        # This keeps the condition simple: v_i·cv <= cv_norm_sq/2
        
        if cv_norm_sq != max_norm_current:
            continue
        
        # Check dot products with all existing vectors
        dots = current @ cv
        # Need v_i·cv <= cv_norm_sq/2 = 1
        if np.max(dots) <= 1:
            current = np.vstack([current, cv])
    
    # Final verification
    if not check_valid(current):
        # Fallback to original
        return vectors
    
    return current

# EVOLVE-BLOCK-END
