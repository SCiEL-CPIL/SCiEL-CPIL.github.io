import numpy as np
import itertools

D = 11

def construct():
    # Generate candidate vectors with various norms
    # We'll focus on:
    # 1. Norm squared 2: two coordinates +/- 1 (220 vectors)
    # 2. Norm squared 4: one coordinate +/- 2 (22 vectors) or four coordinates +/- 1 (not all, but some)
    # 3. Norm squared 8: four coordinates +/- 1 (many vectors)
    
    # Let's think about this differently.
    # The kissing number in dimension 11 is known to be 504 (from tables).
    # But we're restricted to integer vectors.
    # 
    # For integer vectors, the best known constructions use:
    # - Vectors of the form (±1, ±1, 0, ..., 0) with norm √2
    # - Vectors of the form (±2, 0, ..., 0) with norm 2
    # - Vectors of the form (±1, ±1, ±1, ±1, 0, ..., 0) with norm 2
    #
    # Wait, (±1,±1,±1,±1) has norm squared = 4, so norm = 2.
    # 
    # Let me reconsider. If we use only norm-2 vectors (norm squared = 2),
    # the maximum pairwise distance constraint is:
    # min ||u-v|| >= max ||u|| = √2
    # So we need ||u-v|| >= √2, i.e., ||u-v||² >= 2.
    # ||u-v||² = ||u||² + ||v||² - 2u·v = 2 + 2 - 2u·v = 4 - 2u·v
    # So 4 - 2u·v >= 2 => u·v <= 1.
    # 
    # For two vectors of the form (±1,±1,0,...,0), their dot product is:
    # - 2 if they share both coordinates with same signs
    # - 0 if they share one coordinate (one +1, one -1) or share no coordinates but overlap in support
    # - -2 if they share both coordinates with opposite signs
    # - 0 if they have disjoint supports
    # 
    # Actually, let me compute more carefully.
    # u = (a, b, 0, ..., 0) where a,b ∈ {±1}
    # v = (c, d, 0, ..., 0) where c,d ∈ {±1}
    # u·v = a*c + b*d if they share the same pair of coordinates, or 0 if different pairs.
    # 
    # If same pair: u·v = a*c + b*d ∈ {-2, 0, 2}
    # If different pairs: u·v = 0
    # 
    # So we need u·v <= 1, which excludes u·v = 2.
    # u·v = 2 happens when a=c and b=d, i.e., u and v are identical or... wait,
    # if u=(1,1,0,...) and v=(1,1,0,...), they're the same vector.
    # If u=(1,1,0,...) and v=(1,-1,0,...), u·v = 1-1 = 0. OK.
    # If u=(1,1,0,...) and v=(-1,1,0,...), u·v = -1+1 = 0. OK.
    # If u=(1,1,0,...) and v=(-1,-1,0,...), u·v = -1-1 = -2. OK.
    # 
    # So actually, any two distinct vectors of the form (±1,±1,0,...,0) have u·v ≤ 0.
    # Therefore, all 220 such vectors are mutually compatible!
    # 
    # Now, can we add vectors of norm 2 (norm squared = 4)?
    # If we add a vector w with ||w|| = 2, then max norm becomes 2.
    # We need ||u-v|| >= 2 for all pairs, including those involving w.
    # For u (norm √2) and w (norm 2): ||u-w||² = 2 + 4 - 2u·w = 6 - 2u·w >= 4
    # So u·w <= 1.
    # 
    # For w = (2,0,...,0): u·w = 2*u_0. Since u_0 ∈ {-1,0,1}, u·w ∈ {-2,0,2}.
    # We need u·w <= 1, so u_0 must be 0 or -1 (not 1).
    # 
    # This gets complicated. Let's just try to maximize the count.
    # 
    # Strategy: Start with all 220 norm-2 vectors. Check if we can add any norm-4 vectors.
    # If max norm becomes 2, we need all pairwise distances >= 2.
    # 
    # Let's implement a greedy approach:
    # 1. Start with all 220 norm-2 vectors.
    # 2. Try to add norm-4 vectors one by one, checking compatibility.
    # 3. If adding a norm-4 vector forces removal of some norm-2 vectors, do it if net gain is positive.
    
    # First, generate all norm-2 vectors (norm squared = 2)
    norm2_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                norm2_vectors.append(v)
    
    norm2_vectors = np.array(norm2_vectors)
    
    # Generate norm-4 vectors (norm squared = 4)
    # Type 1: one coordinate is ±2
    norm4_vectors = []
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            norm4_vectors.append(v)
    
    # Type 2: four coordinates are ±1
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for c, s in zip(coords, signs):
                v[c] = s
            norm4_vectors.append(v)
    
    norm4_vectors = np.array(norm4_vectors)
    
    # Greedy: start with all norm-2 vectors
    selected = list(norm2_vectors)
    max_norm_sq = 2
    
    # Try to add norm-4 vectors
    # For each candidate norm-4 vector, check if it's compatible with all selected
    # and if adding it (and possibly removing incompatible norm-2 vectors) is beneficial
    
    def is_compatible(u, v, max_norm_sq):
        """Check if u and v are compatible given max_norm_sq."""
        dist_sq = np.sum((u - v) ** 2)
        return dist_sq >= max_norm_sq
    
    # For simplicity, let's just try adding norm-4 vectors without removing norm-2 vectors
    # first, and see if we can improve.
    
    improved = False
    for w in norm4_vectors:
        # Check if w is compatible with all currently selected vectors
        compatible = True
        for u in selected:
            if not is_compatible(u, w, max_norm_sq):
                compatible = False
                break
        if compatible:
            selected.append(w)
            max_norm_sq = 4  # Since ||w||^2 = 4
            improved = True
            # After adding a norm-4 vector, we need to re-check all pairs
            # because the max norm changed. This is expensive, so let's break early
            # and do a full re-validation.
            break
    
    # If we added a norm-4 vector, we need to re-validate the entire set
    # because the distance constraint changed from √2 to 2.
    if improved:
        # Re-validate: remove any vector that violates the new constraint
        new_selected = []
        for i, u in enumerate(selected):
            valid = True
            for j, v in enumerate(selected):
                if i == j:
                    continue
                if not is_compatible(u, v, max_norm_sq):
                    valid = False
                    break
            if valid:
                new_selected.append(u)
        selected = new_selected
    
    # Convert to numpy array
    if len(selected) > 0:
        result = np.array(selected)
    else:
        result = norm2_vectors
    
    # Final validation
    if len(result) >= 2:
        norms_sq = np.sum(result ** 2, axis=1)
        max_norm_sq = np.max(norms_sq)
        # Compute pairwise distances
        dists_sq = np.sum((result[:, np.newaxis, :] - result[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = np.min(dists_sq)
        if min_dist_sq < max_norm_sq:
            # Invalid, fall back to norm-2 only
            result = norm2_vectors
    
    return result
