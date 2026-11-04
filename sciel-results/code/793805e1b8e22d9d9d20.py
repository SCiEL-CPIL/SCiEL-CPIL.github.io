import numpy as np
import itertools
import time

def construct():
    D = 11
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Generate all pairs of indices (supports of size 2)
    supports = list(itertools.combinations(range(D), 2))
    n = len(supports)  # 55 supports
    
    # For each support, we have 4 sign combinations: (1,1), (1,-1), (-1,1), (-1,-1)
    # Two vectors conflict if their dot product is 0 (perpendicular) OR if they are the same support
    # Actually, the constraint is: minimum pairwise distance >= maximum norm
    # For vectors of norm sqrt(2), distance between u and v is sqrt(2 + 2 - 2*u.v) = sqrt(4 - 2*u.v)
    # We need sqrt(4 - 2*u.v) >= sqrt(2) for all pairs, so 4 - 2*u.v >= 2, i.e., u.v <= 1
    # Since vectors have entries in {-1, 0, 1} with exactly two non-zero entries, u.v can be -2, -1, 0, 1, 2
    # We need u.v <= 1, so we CANNOT have u.v = 2 (which means u = v, same vector)
    # Also u.v = -2 means u = -v, which gives distance sqrt(4+2) = sqrt(6) > sqrt(2), so that's fine
    # Wait, let me reconsider. The condition is min pairwise distance >= max norm.
    # All vectors have norm sqrt(2), so max norm = sqrt(2).
    # Distance between u and v: ||u - v||^2 = ||u||^2 + ||v||^2 - 2u.v = 2 + 2 - 2u.v = 4 - 2u.v
    # We need ||u - v|| >= sqrt(2), so 4 - 2u.v >= 2, i.e., u.v <= 1.
    # u.v = 2 only if u = v (same vector), which we exclude by having unique vectors.
    # u.v = 1: e.g., u = (1,1,0...), v = (1,0,1,0...) -> u.v = 1. Distance = sqrt(2). OK.
    # u.v = 0: perpendicular. Distance = sqrt(4) = 2. OK.
    # u.v = -1: Distance = sqrt(6). OK.
    # u.v = -2: u = -v. Distance = sqrt(8) = 2*sqrt(2). OK.
    # So actually ALL distinct vectors with exactly two non-zero entries of magnitude 1 are valid!
    # The only conflict is u = v (same vector), which we avoid by construction.
    
    # Wait, but the current program already generates all 55*4 = 220 vectors and gets score 1.0.
    # The inspirations mention scores of 2.5455, which would be 220 * 2.5455 ≈ 560 vectors.
    # So we need to use vectors with more non-zero entries or different norms.
    
    # Let me reconsider. We can use vectors of different norms. The condition is:
    # min_{u != v} ||u - v|| >= max_u ||u||
    # This is a more general condition. Let's try using vectors with entries in {-1, 0, 1} 
    # but with varying numbers of non-zero entries.
    
    # Actually, let's think about this differently. The kissing number problem in dimension 11.
    # The known kissing number in dimension 11 is 440 (or nearby).
    # A common construction uses the Leech lattice or similar.
    
    # Let me try a different approach: use all vectors with exactly 2 non-zero entries (±1),
    # AND also vectors with exactly 1 non-zero entry (±1) if they don't conflict.
    
    # Vector with 1 non-zero entry: norm = 1. Vector with 2 non-zero entries: norm = sqrt(2).
    # Max norm = sqrt(2). Distance between (1,0,...) and (1,1,0,...) = sqrt(1+1-1) = 1 < sqrt(2). CONFLICT!
    # So we can't mix different support sizes easily.
    
    # Let me reconsider the problem. Perhaps we should use vectors of the same norm.
    # If all vectors have norm 2 (entries in {-2, 0, 2} or {-1, 0, 1} scaled), then:
    # ||u - v||^2 = 4 + 4 - 2u.v = 8 - 2u.v. Need 8 - 2u.v >= 4, i.e., u.v <= 2.
    
    # For vectors with entries in {-1, 0, 1} and exactly 4 non-zero entries: norm = 2.
    # u.v can range from -4 to 4. We need u.v <= 2.
    # This is a more interesting constraint!
    
    # Let's generate all vectors with exactly 4 non-zero entries of magnitude 1, 
    # and find the maximum set where all pairwise dot products <= 2.
    
    # Number of such vectors: C(11,4) * 2^4 = 330 * 16 = 5280
    # This is large but manageable for a conflict graph approach.
    
    # Generate all supports of size 4
    supports4 = list(itertools.combinations(range(D), 4))
    n_supports = len(supports4)  # 330
    
    # For each support, there are 16 sign combinations
    # Total vectors: 330 * 16 = 5280
    
    # Conflict: two vectors u, v conflict if u.v > 2
    # u.v > 2 means u.v >= 3 (since dot products are integers)
    # For vectors with 4 non-zero entries each, u.v = sum of products of corresponding entries
    # u.v = 3 or 4 means they share at least 3 non-zero positions with matching signs
    
    # Let's build the conflict graph using bitmasks for efficiency
    
    # Represent each support as a bitmask of 11 bits
    support_masks = []
    for sup in supports4:
        mask = 0
        for idx in sup:
            mask |= (1 << idx)
        support_masks.append(mask)
    
    # Two vectors with supports A and B and signs s_A, s_B have dot product:
    # u.v = sum_{i in A∩B} s_A[i] * s_B[i]
    # |u.v| <= |A∩B|
    # u.v > 2 iff u.v >= 3, which requires |A∩B| >= 3 and the signs align enough
    
    # For a given pair of supports (A, B), the maximum possible dot product is |A∩B|.
    # We need to check if there exist sign assignments where u.v > 2.
    # Actually, we need: for the specific sign assignments, u.v <= 2.
    
    # This is complex. Let me use a different approach.
    
    # Alternative: Use vectors of norm 2 with entries in {-1, 0, 1}, exactly 4 non-zeros.
    # Build the full conflict graph and solve Maximum Independent Set.
    
    # But 5280 nodes is too many for exact MIS in reasonable time.
    
    # Let me try a smaller construction. Use vectors with exactly 3 non-zero entries.
    # Norm = sqrt(3). Need ||u-v||^2 >= 3, so 3+3-2u.v >= 3, u.v <= 1.5, so u.v <= 1.
    # u.v for two vectors with 3 non-zeros each: max |A∩B| = 3.
    # u.v > 1 means u.v >= 2, requires |A∩B| >= 2 with aligned signs.
    
    # Number of vectors: C(11,3) * 2^3 = 165 * 8 = 1320
    # Still large but more manageable.
    
    # Let me try the branch-and-bound approach on the support level first.
    # Two supports conflict if they share >= 2 indices (for the 3-nonzero case, need to check signs too).
    
    # Actually, let me just go with the 2-nonzero vectors but add more structure.
    # The current 220 vectors give score 1.0. To get 2.5x, we need ~550 vectors.
    
    # Let me try: all vectors with exactly 2 non-zero entries (±1) = 220 vectors.
    # Plus: vectors with exactly 4 non-zero entries that are "compatible" with all of them.
    # A vector v with 4 non-zeros conflicts with a 2-nonzero vector u if u.v > 1 (for norm sqrt(2) max).
    # Wait, max norm would be max(sqrt(2), 2) = 2. So we need ||u-v|| >= 2.
    # ||u-v||^2 = 2 + 4 - 2u.v = 6 - 2u.v >= 4, so u.v <= 1.
    # For u with 2 non-zeros and v with 4 non-zeros, u.v = sum of products on shared indices.
    # If they share 2 indices: u.v can be -2, 0, 2. We need u.v <= 1, so u.v != 2.
    # If they share 1 index: u.v can be -1, 1. Always <= 1. OK.
    # If they share 0 indices: u.v = 0. OK.
    
    # So a 4-nonzero vector v is compatible with all 2-nonzero vectors iff:
    # For every pair of indices in v's support, there's no 2-nonzero vector with the same signs.
    # But we're including ALL 2-nonzero vectors, so for any pair (i,j) in v's support,
    # the vector with +1 at i and +1 at j exists, and its dot product with v is 2 if v[i]=v[j]=1.
    # So v conflicts with all 2-nonzero vectors if it has any pair with same sign.
    # To avoid conflict, v must have all different signs on its 4 positions... but there are only 2 signs!
    # So with 4 positions and 2 signs, by pigeonhole, at least 2 have the same sign.
    # Thus, ANY 4-nonzero vector conflicts with some 2-nonzero vector.
    
    # This means we can't simply add 4-nonzero vectors to the 2-nonzero set.
    
    # New approach: Use ONLY vectors with the same number of non-zero entries.
    # Let's use 4 non-zero entries, norm = 2, and find max set with pairwise dot products <= 2.
    
    # Build conflict graph on 5280 vectors. Two vectors conflict if dot > 2.
    # Use bitmasks for supports and sign patterns.
    
    # For speed, let's precompute for each support pair the overlap and check sign compatibility.
    
    # Actually, let me try a greedy approach with local search on the 5280-vector graph.
    
    # Generate all 5280 vectors
    all_vectors = []
    vector_supports = []
    for sup in supports4:
        mask = 0
        for idx in sup:
            mask |= (1 << idx)
        vector_supports.append(mask)
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            all_vectors.append(v)
    
    n_vec = len(all_vectors)  # 5280
    
    # Build conflict matrix using dot products
    # Two vectors conflict if dot > 2
    V = np.array(all_vectors, dtype=np.int8)
    
    # Compute all pairwise dot products
    # This is 5280 x 5280 = ~28M entries, feasible
    dots = V @ V.T  # shape (5280, 5280)
    
    # Conflict if dot > 2 (and i != j)
    conflict = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Greedy independent set
    selected = np.zeros(n_vec, dtype=bool)
    remaining = set(range(n_vec))
    
    # Sort by degree (number of conflicts) - prefer low degree nodes
    degrees = conflict.sum(axis=1)
    order = np.argsort(degrees)
    
    for idx in order:
        if time.time() > deadline:
            break
        if not remaining.__contains__(int(idx)):
            continue
        # Check if idx conflicts with any already selected
        if selected[idx].any() if hasattr(selected[idx], 'any') else False:
            continue
        # Actually, let's check properly
        # selected is a boolean array, we need to check if any selected node conflicts with idx
        if np.any(conflict[idx] & selected):
            continue
        selected[idx] = True
        remaining.discard(int(idx))
        # Remove all neighbors of idx from remaining
        neighbors = np.where(conflict[idx] == 1)[0]
        for nb in neighbors:
            remaining.discard(int(nb))
    
    # Local search: try to add more vectors
    # Try swapping: remove one selected, add two non-selected
    improved = True
    while improved and time.time() < deadline:
        improved = False
        selected_indices = np.where(selected)[0]
        non_selected = np.where(~selected)[0]
        
        for i in range(len(selected_indices)):
            if time.time() > deadline:
                break
            rem = selected_indices[i]
            # Temporarily remove rem
            selected[rem] = False
            # Try to add non-selected nodes that don't conflict with current selected
            for j in range(len(non_selected)):
                if time.time() > deadline:
                    break
                cand = non_selected[j]
                if selected[cand]:
                    continue
                if np.any(conflict[cand] & selected):
                    continue
                # Try to add cand
                selected[cand] = True
                # Try to add one more
                for k in range(j+1, len(non_selected)):
                    if time.time() > deadline:
                        break
                    cand2 = non_selected[k]
                    if selected[cand2]:
                        continue
                    if np.any(conflict[cand2] & selected):
                        continue
                    selected[cand2] = True
                    # We added 2, removed 1, net +1
                    improved = True
                    break
                if improved:
                    break
            if not improved:
                # Restore rem
                selected[rem] = True
        
    result = V[selected]
    
    if len(result) == 0:
        # Fallback to the simple 220-vector solution
        vectors = []
        for support in itertools.combinations(range(D), 2):
            i, j = support
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = s1
                    v[j] = s2
                    vectors.append(v)
        return np.array(vectors, dtype=np.int8)
    
    return result.astype(np.int8)
