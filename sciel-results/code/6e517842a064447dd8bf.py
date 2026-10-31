import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Strategy: Generate a large pool of integer vectors in R^11 with the same norm^2,
    # then find a large subset where all pairwise distances are at least the max norm.
    # 
    # Key insight: If all vectors have norm^2 = R^2, then dist^2(i,j) = 2*R^2 - 2*v_i·v_j.
    # We need dist^2 >= R^2, i.e., 2*R^2 - 2*v_i·v_j >= R^2, i.e., v_i·v_j <= R^2/2.
    #
    # Let's use norm^2 = 4 (so vectors have entries in {-1, 0, 1} with exactly 4 nonzero entries,
    # or weight-2 vectors scaled by 2, etc.)
    #
    # Actually, let's think more carefully. The kissing number in R^11 is known to be at least 554.
    # The Leech lattice approach or other constructions can get us there.
    #
    # Let me use a different approach: generate many candidate vectors and use a greedy 
    # maximal independent set approach on the conflict graph.
    #
    # Pool generation:
    # 1. Weight-2 vectors in {-2, 0, 2}^11: norm^2 = 8. Count: C(11,2)*4 = 220
    # 2. Weight-4 vectors in {-1, 0, 1}^11: norm^2 = 4. Count: C(11,4)*16 = 330*16 = 5280
    #    But these have different norm than group 1.
    #
    # Let's use all weight-4 vectors in {-1,0,1}^11 (norm^2 = 4):
    # Constraint: v_i·v_j <= 2
    # Two weight-4 vectors with dot product <= 2 are valid.
    #
    # Alternatively, use weight-2 vectors scaled to norm^2=8: entries in {-2,0,2}
    # Constraint: v_i·v_j <= 4
    #
    # Let me try a mixed approach with norm^2 = 4:
    # Weight-4 vectors in {-1,0,1}^11: C(11,4)*2^4 = 330*16 = 5280 vectors
    # For two such vectors, dot product ranges from -4 to 4.
    # We need dot <= 2.
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    vectors = np.array(candidates, dtype=np.int64)
    N = len(vectors)
    
    if N == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Compute norms
    norms2 = (vectors ** 2).sum(axis=1)
    max_norm2 = norms2.max()
    
    # Compute dot products
    dots = vectors @ vectors.T
    
    # dist^2(i,j) = norms2[i] + norms2[j] - 2*dots[i,j]
    dist2 = norms2[:, None] + norms2[None, :] - 2 * dots
    np.fill_diagonal(dist2, 0)
    
    min_dist2 = dist2.min()
    
    if min_dist2 >= max_norm2:
        # All vectors are valid together
        return vectors
    
    # Need to find a large subset with min dist^2 >= max_norm2
    # Conflict: dist2[i,j] < max_norm2
    conflicts = dist2 < max_norm2
    np.fill_diagonal(conflicts, False)
    
    # Greedy MIS with minimum degree heuristic
    alive = np.ones(N, dtype=bool)
    selected = []
    degrees = conflicts.sum(axis=1).astype(np.int32)
    rng = np.random.default_rng(42)
    
    while np.any(alive) and time.time() < deadline:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        selected.append(chosen)
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        alive[to_remove] = False
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
    
    result_vectors = vectors[selected]
    return result_vectors
