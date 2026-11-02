"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Start with the base set: vectors with exactly two ±1 entries
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                vectors.append(v)
    
    # Also add vectors with a single ±1 entry (norm 1)
    # Wait - if we add norm-1 vectors, the max norm becomes 2, and we need distance >= 2
    # A norm-1 vector and a norm-2 vector: need ||v1 - v2|| >= 2
    # ||v1 - v2||^2 = 1 + 4 - 2*v1.v2 = 5 - 2*v1.v2 >= 4 => v1.v2 <= 0.5 => v1.v2 <= 0
    # So norm-1 vector must be orthogonal to all norm-2 vectors? That's too restrictive.
    # Actually let's just use vectors all with the same norm to simplify.
    
    # Try vectors with three non-zero entries: ±1 in three positions, norm = sqrt(3)
    # Or keep norm 2 but try different patterns
    
    # Greedy: try to add more vectors with norm^2 = 2 (two ±1 entries) - already done
    # Try vectors with norm^2 = 1 (one ±1 entry)
    # For all current vectors (norm^2=2), a new vector with norm^2=1 needs:
    # ||new - old||^2 = 1 + 2 - 2*dot >= 2 (since max norm is sqrt(2), so min dist >= sqrt(2))
    # Wait, the constraint is min pairwise distance >= max norm
    # If max norm is sqrt(2), then min pairwise dist >= sqrt(2)
    # ||a-b||^2 >= 2
    # For a with norm^2=2 and b with norm^2=1: ||a-b||^2 = 2+1-2a.b = 3-2a.b >= 2 => a.b <= 0.5 => a.b <= 0
    # So b must be orthogonal to all a. But a spans all pairs, so b must be 0. Not possible.
    
    # Let's try a different approach: use all vectors with entries in {-1, 0, 1} and norm^2 = 2
    # Already have all such vectors.
    
    # Try adding vectors with larger norms but ensure constraint holds
    # Or use a completely different construction
    
    # Known: kissing number in 11D is at least 504 (from Leech lattice related constructions)
    # Let's try a greedy local search approach
    
    # Convert to numpy array for efficient computation
    vecs = np.array(vectors, dtype=np.float64)
    n = len(vecs)
    
    # Compute all pairwise distances and max norm
    # For efficiency, use matrix operations
    norms_sq = np.sum(vecs * vecs, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Try to add new candidate vectors
    # Generate candidates with entries in {-2, -1, 0, 1, 2}
    # Focus on vectors with small support
    
    added = 0
    candidates = []
    
    # Try vectors with exactly 3 non-zero entries from {-1, 1}
    # norm^2 = 3, so max norm becomes sqrt(3)
    # Need pairwise distances >= sqrt(3)
    # Between two such vectors: ||a-b||^2 >= 3
    # a.b <= (3+3-3)/2 = 1.5 => a.b <= 1
    # Between new (norm^2=3) and old (norm^2=2): ||a-b||^2 = 3+2-2a.b >= 3 => a.b <= 1
    # So dot product <= 1 for all pairs
    
    # This changes the max norm, so we need to recheck all pairs.
    # Let's try a simpler approach: keep all vectors at norm^2 = 2
    
    # Alternative: use a different known construction
    # The 24-cell and other polytopes...
    
    # Let's try a greedy approach with random vectors
    rng = np.random.RandomState(42)
    
    # Try adding vectors with entries in {-1, 0, 1} that have norm^2 = 2
    # We already have all of those.
    
    # Try vectors with one entry being 2 and others 0: norm = 2
    # max norm would be 2, need pairwise dist >= 2
    # Two such vectors: ||2e_i - 2e_j|| = 2*sqrt(2) >= 2 ✓
    # 2e_i vs existing vector with entries at i,j: ||2e_i - (e_i+e_j)||^2 = (2-1)^2 + (0-1)^2 = 2 >= 4? No, 2 < 4.
    # So this doesn't work with mixed norms.
    
    # Let me reconsider. Keep all vectors at the same norm for simplicity.
    # Current: 220 vectors, all norm^2 = 2, max norm = sqrt(2)
    # Min pairwise dist: for two vectors sharing one coordinate: 
    # e_i+e_j vs e_i+e_k: diff = e_j - e_k, norm = sqrt(2) >= sqrt(2) ✓
    # e_i+e_j vs e_i-e_j: diff = 2e_j, norm = 2 >= sqrt(2) ✓
    # e_i+e_j vs e_k+e_l (disjoint): diff norm = sqrt(4) = 2 >= sqrt(2) ✓
    # So all valid.
    
    # Can we add more vectors with norm^2 = 2? No, we have all C(11,2)*4 = 220.
    
    # Try norm^2 = 3 vectors (three ±1 entries)
    # If we add these, max norm = sqrt(3), need all pairwise dists >= sqrt(3)
    # Check old-old: min dist was sqrt(2) < sqrt(3). FAIL.
    # So we can't mix.
    
    # We need a homogeneous construction or a different approach.
    
    # Let's try: use only vectors with norm^2 = 3 (three ±1 entries)
    # Count: C(11,3) * 8 = 165 * 8 = 1320 vectors
    # Need pairwise dist >= sqrt(3)
    # Two vectors with 3 non-zero entries:
    # If they share 2 positions: e.g., (1,1,1,0,...) and (1,1,-1,0,...)
    # diff = (0,0,2,0,...), norm = 2 >= sqrt(3) ✓
    # If they share 1 position: (1,1,1,0,...) and (1,0,0,1,...)
    # diff = (0,1,1,-1,0,...), norm = sqrt(3) >= sqrt(3) ✓
    # If they share 0 positions: diff norm = sqrt(6) >= sqrt(3) ✓
    # If same support, same signs: identical, not allowed (nonzero distinct)
    # If same support, differ in 1 sign: diff has 2 entries of ±2, norm = 2*sqrt(2) >= sqrt(3) ✓
    # If same support, differ in 2 signs: diff has 4 entries, norm = 2*sqrt(4)... 
    # (1,1,1) vs (1,-1,-1): diff = (0,2,2), norm = sqrt(8) >= sqrt(3) ✓
    # If same support, differ in 3 signs: (1,1,1) vs (-1,-1,-1): diff = (2,2,2), norm = sqrt(12) ✓
    
    # Wait, I need to check more carefully. Let a and b be two distinct vectors with entries in {-1,1} on 3 positions.
    # ||a-b||^2 = sum over all positions of (a_i - b_i)^2
    # For positions where both are 0: contributes 0
    # For positions where one is 0 and other is ±1: contributes 1
    # For positions where both are ±1: contributes 0 if same, 4 if different
    
    # Case: same support (same 3 positions)
    # Let k = number of positions where signs differ, k ∈ {1,2,3}
    # ||a-b||^2 = 4k
    # k=1: 4 >= 3 ✓
    # k=2: 8 >= 3 ✓  
    # k=3: 12 >= 3 ✓
    
    # Case: supports overlap in 2 positions
    # a has entries on {i,j,k}, b has entries on {i,j,l} with l≠k
    # Positions i,j: both have ±1, contribute 0 or 4 each
    # Position k: a has ±1, b has 0, contribute 1
    # Position l: a has 0, b has ±1, contribute 1
    # Min ||a-b||^2 = 0 + 0 + 1 + 1 = 2 < 3 ✗ FAIL!
    
    # Example: a = (1,1,1,0,...), b = (1,1,0,1,...)
    # diff = (0,0,1,-1,0,...), norm = sqrt(2) < sqrt(3). FAIL.
    
    # So we can't just take all norm^2=3 vectors.
    
    # We need to select a subset. Let's use a greedy approach.
    
    # Generate all candidates with norm^2 = 3
    all_cands = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([-1, 1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(support, signs):
                v[pos] = sign
            all_cands.append(v)
    
    all_cands = np.array(all_cands, dtype=np.float64)
    n_cands = len(all_cands)
    
    # Greedy: start with empty set, add vectors that satisfy constraint with all existing
    # For efficiency, maintain a list of accepted vectors
    accepted = []
    max_norm_sq = 3.0  # all have norm^2 = 3
    
    # For each candidate, check if it's at distance >= sqrt(3) from all accepted
    # ||a-b||^2 >= 3
    
    # Use a bitset or just iterate
    # For speed, precompute candidate norms (all 3)
    
    # Greedy with random ordering for diversity
    indices = rng.permutation(n_cands)
    
    for idx in indices:
        if time.time() > deadline - 5:
            break
        v = all_cands[idx]
        # Check against all accepted
        if len(accepted) == 0:
            accepted.append(v)
            continue
        # Compute distances to all accepted
        # ||v - a||^2 = ||v||^2 + ||a||^2 - 2*v.a = 6 - 2*v.a
        # Need >= 3 => v.a <= 1.5 => v.a <= 1 (integer)
        dots = np.dot(np.array(accepted), v)
        if np.all(dots <= 1):
            accepted.append(v)
            added += 1
    
    if len(accepted) > 220:
        return np.array(accepted, dtype=np.int64)
    
    # Fallback: return original
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
