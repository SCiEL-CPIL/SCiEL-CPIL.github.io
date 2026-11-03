"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # The seed program uses vectors with exactly two non-zero entries (±1, ±1).
    # There are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    # All have norm sqrt(2). The minimum pairwise distance among them:
    # Two such vectors v, w have distance squared = ||v-w||^2 = ||v||^2 + ||w||^2 - 2<v,w>
    # = 2 + 2 - 2<v,w> = 4 - 2<v,w>.
    # For the constraint "min pairwise distance >= max norm", we need:
    # 4 - 2<v,w> >= sqrt(2) for all pairs.
    # The worst case is when <v,w> is maximized. Two weight-2 vectors can share
    # one coordinate (dot product = 1) or be disjoint (dot product = 0), or be
    # identical (dot product = 2, but we don't include duplicates).
    # If they share one coordinate with same sign: dot = 1, dist^2 = 2, dist = sqrt(2).
    # If they share one coordinate with opposite sign: dot = -1, dist^2 = 6, dist = sqrt(6).
    # If disjoint: dot = 0, dist^2 = 4, dist = 2.
    # So the minimum distance is sqrt(2) = max norm. This is valid!
    
    # Can we add more vectors? Let's try vectors with weight 3 (three ±1 entries).
    # Norm = sqrt(3). For the constraint: min dist >= sqrt(3).
    # But we already have vectors of norm sqrt(2) < sqrt(3), so max norm would be sqrt(3).
    # A weight-2 vector v and weight-3 vector w: dist^2 = 2 + 3 - 2<v,w> = 5 - 2<v,w>.
    # We need 5 - 2<v,w> >= 3, i.e., <v,w> <= 1.
    # <v,w> can be at most 2 (if v's two non-zeros are both in w with same sign).
    # So we need to avoid pairs where <v,w> = 2.
    
    # Let's try a greedy approach: start with all weight-2 vectors, then try adding
    # weight-3 vectors that are compatible.
    
    # First, generate all weight-2 vectors
    from itertools import combinations, product
    
    weight2 = []
    for i, j in combinations(range(D), 2):
        for si, sj in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            weight2.append(v)
    
    weight2 = np.array(weight2, dtype=np.int64)
    
    # Generate all weight-3 vectors
    weight3 = []
    for i, j, k in combinations(range(D), 3):
        for si, sj, sk in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            v[k] = sk
            weight3.append(v)
    
    weight3 = np.array(weight3, dtype=np.int64)
    
    # Check compatibility: for each weight-3 vector, check against all weight-2 vectors
    # <v,w> <= 1 for all v in weight3, w in weight2
    # Also need weight-3 vectors to be compatible with each other:
    # dist^2 = 3 + 3 - 2<v,w> = 6 - 2<v,w> >= 3, so <v,w> <= 1.5, i.e., <v,w> <= 1.
    
    # Compute dot products between weight3 and weight2
    # This is a 165 x 220 matrix
    dots_w3_w2 = weight3 @ weight2.T  # (165, 220)
    
    # A weight-3 vector is compatible with all weight-2 if max dot <= 1
    w3_compatible_with_w2 = np.all(dots_w3_w2 <= 1, axis=1)
    
    # Now among compatible weight-3 vectors, find a subset where pairwise dot <= 1
    # This is a maximum independent set problem on a graph where edges connect
    # weight-3 vectors with dot > 1.
    
    # Let's use a greedy approach with multiple random orderings
    candidate_indices = np.where(w3_compatible_with_w2)[0]
    n_candidates = len(candidate_indices)
    
    if n_candidates > 0:
        # Compute pairwise dots among candidates
        cand_vecs = weight3[candidate_indices]
        dots_cand = cand_vecs @ cand_vecs.T  # (n, n)
        
        # Greedy: try several random orderings, pick max independent set
        best_w3_subset = []
        
        # Deterministic greedy first
        order = np.argsort(-np.sum(dots_cand, axis=1))  # most connected first
        selected = []
        for idx in order:
            if len(selected) == 0:
                selected.append(idx)
            else:
                # Check dot with all selected
                if np.all(dots_cand[idx, selected] <= 1):
                    selected.append(idx)
        
        best_w3_subset = selected
        
        # Try a few random orderings
        rng = np.random.default_rng(42)
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            order = rng.permutation(n_candidates)
            selected = []
            for idx in order:
                if len(selected) == 0:
                    selected.append(idx)
                else:
                    if np.all(dots_cand[idx, selected] <= 1):
                        selected.append(idx)
            if len(selected) > len(best_w3_subset):
                best_w3_subset = selected
        
        # Add the selected weight-3 vectors
        if best_w3_subset:
            additional = cand_vecs[best_w3_subset]
            all_vectors = np.vstack([weight2, additional])
        else:
            all_vectors = weight2
    else:
        all_vectors = weight2
    
    # Final verification
    if len(all_vectors) == 0:
        return weight2
    
    norms = np.sqrt(np.sum(all_vectors.astype(float)**2, axis=1))
    max_norm = np.max(norms)
    
    # Check min pairwise distance
    # For large sets this is O(n^2 * D), but n is at most ~400 so it's fine
    n = len(all_vectors)
    min_dist = np.inf
    for i in range(n):
        if time.time() > deadline - 5:
            break
        diffs = all_vectors[i+1:] - all_vectors[i]
        dists = np.sqrt(np.sum(diffs.astype(float)**2, axis=1))
        if len(dists) > 0:
            min_dist = min(min_dist, np.min(dists))
    
    if min_dist >= max_norm - 1e-9:
        return all_vectors.astype(np.int64)
    else:
        # Fallback to weight-2 only
        return weight2

# EVOLVE-BLOCK-END
