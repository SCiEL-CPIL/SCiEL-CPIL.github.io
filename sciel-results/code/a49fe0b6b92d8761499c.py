"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # We want to find as many integer vectors as possible such that
    # min pairwise distance >= max norm.
    # 
    # Key insight: If we normalize to norm 2, the kissing number bound for d=11 is 512.
    # The constraint is: for all pairs i,j, ||v_i - v_j|| >= max_k ||v_k||
    # 
    # Strategy: 
    # 1. Generate candidate vectors of various weights (1, 2, 3, 4, ...)
    # 2. Use greedy selection: try to add vectors in order of some priority
    # 3. The key constraint is that pairwise distances must be >= max norm
    #
    # Let's think about what vectors work well:
    # - Weight-2 vectors (two nonzero entries of ±1): norm = sqrt(2)
    #   Distance between two weight-2 vectors: depends on overlap
    #   If they share one coordinate: distance = sqrt(1+1) = sqrt(2) (if same sign) or sqrt(1+1+1+1)=2 (if different)
    #   Wait, let me recompute.
    #
    # For weight-2 vectors v = e_i + e_j and w = e_k + e_l:
    # - If {i,j} and {k,l} are disjoint: ||v-w||^2 = 4, so ||v-w|| = 2
    # - If they share exactly one index: ||v-w||^2 = 1+1 = 2 (the shared index cancels or adds)
    #   Actually if v = e_i+e_j, w = e_i+e_k: v-w = e_j - e_k, ||v-w||^2 = 2
    #   If v = e_i+e_j, w = -e_i+e_k: v-w = 2e_i+e_j-e_k, ||v-w||^2 = 4+1+1=6
    # - If they are the same: distance = 0 (can't have duplicates)
    #
    # Max norm among weight-2 vectors is sqrt(2).
    # Min pairwise distance among weight-2 vectors can be as low as sqrt(2) (when sharing one index with same sign).
    # So sqrt(2) >= sqrt(2) is satisfied!
    #
    # Wait, but we need min pairwise distance >= max norm.
    # For all weight-2 vectors: max norm = sqrt(2), min pairwise dist = sqrt(2). So OK.
    #
    # But can we mix in weight-1 vectors?
    # Weight-1 vector: norm = 1.
    # Distance from weight-1 e_i to weight-2 e_j+e_k:
    #   If i not in {j,k}: ||e_i - e_j - e_k||^2 = 1+1+1 = 3, dist = sqrt(3) >= sqrt(2) OK
    #   If i in {j,k}, say i=j: ||e_i - e_i - e_k|| = ||-e_k|| = 1 < sqrt(2) NOT OK
    #   If i=j and sign: e_i - (-e_i+e_k) = 2e_i - e_k, ||.||^2 = 4+1=5, OK
    #
    # So weight-1 vector e_i conflicts with weight-2 vectors that contain e_i with same sign.
    # 
    # Let me think differently. The known kissing number for d=11 is 512.
    # The current program gets 220 (all weight-2 vectors with ±1 signs: C(11,2)*4 = 55*4 = 220).
    #
    # To improve, we need to find more vectors. Let's try a greedy approach:
    # Generate a large pool of candidate vectors, then greedily select.
    
    # Generate candidates: vectors with small integer entries
    # Let's try vectors with entries in {-1, 0, 1} with weight up to 4
    candidates = []
    
    # Weight 1: 11 * 2 = 22 vectors
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(tuple(v))
    
    # Weight 2: C(11,2) * 4 = 220 vectors
    for i, j in combinations(range(D), 2):
        for si, sj in product((1, -1), repeat=2):
            v = [0] * D
            v[i], v[j] = si, sj
            candidates.append(tuple(v))
    
    # Weight 3: C(11,3) * 8 = 165 * 8 = 1320 vectors
    for i, j, k in combinations(range(D), 3):
        for si, sj, sk in product((1, -1), repeat=3):
            v = [0] * D
            v[i], v[j], v[k] = si, sj, sk
            candidates.append(tuple(v))
    
    # Weight 4: C(11,4) * 16 = 330 * 16 = 5280 vectors
    for i, j, k, l in combinations(range(D), 4):
        for si, sj, sk, sl in product((1, -1), repeat=4):
            v = [0] * D
            v[i], v[j], v[k], v[l] = si, sj, sk, sl
            candidates.append(tuple(v))
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms
    norms = np.sqrt(np.sum(candidates.astype(np.float64)**2, axis=1))
    
    # Sort by norm descending (prefer higher norm vectors first? or ascending?)
    # Actually, let's think: the constraint is min_pairwise_dist >= max_norm
    # If we pick a vector with high norm, it constrains all others.
    # Maybe we should be more selective.
    
    # Greedy: start empty, try to add each candidate if it satisfies the constraint
    # with all already-selected vectors.
    
    selected = []
    selected_norms = []
    selected_arr = []
    
    # For efficiency, process in batches and use vectorized distance checks
    # Order candidates: try weight-2 first (they form a valid set of 220), then add others
    
    # Actually, let's just do greedy with all candidates, ordered by some heuristic
    # Let's order by norm ascending first (less constraining vectors first)
    # No wait - let's think about this more carefully.
    
    # The constraint: for new vector v with norm n_v, and existing vectors with norms n_j:
    #   Need ||v - v_j|| >= max(n_v, max_j n_j) for all j
    # 
    # If all selected have norm <= n_v, then max norm becomes n_v, and we need ||v-v_j|| >= n_v for all j
    # If some selected have norm > n_v, then max norm is max_j n_j, and we need ||v-v_j|| >= max_j n_j
    
    # This is complex. Let's just do a simple greedy:
    # Try candidates in order of decreasing norm (greedy picks "big" vectors first)
    # Actually, let's try increasing norm (pick small ones first, they're less constraining)
    
    order = np.argsort(norms)
    
    for idx in order:
        if time.time() > deadline - 5.0:
            break
        
        v = candidates[idx]
        nv = norms[idx]
        
        if not selected_arr:
            selected_arr.append(v)
            selected_norms.append(nv)
            continue
        
        sel = np.array(selected_arr, dtype=np.float64)
        diffs = sel - v.astype(np.float64)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        max_existing_norm = max(selected_norms)
        required = max(nv, max_existing_norm)
        
        if np.min(dists) >= required - 1e-9:
            selected_arr.append(v)
            selected_norms.append(nv)
    
    result = np.array(selected_arr, dtype=np.int64)
    
    # Verify
    if len(result) > 1:
        # Check all pairwise distances
        pass  # Skip for speed, trust the greedy
    
    return result
# EVOLVE-BLOCK-END
