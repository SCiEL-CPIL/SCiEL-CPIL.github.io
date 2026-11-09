# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = 595.0  # safety margin before 630s limit (evaluator kills at 630s)
    
    def elapsed():
        return time.time() - start_time
    
    # Strategy: Build a large set of integer vectors in R^11 such that
    # min pairwise squared distance >= max squared norm.
    # 
    # Key insight: If all vectors have the same norm^2 = N, then we need
    # ||v_i - v_j||^2 >= N for all i != j.
    # ||v_i - v_j||^2 = 2N - 2*v_i.v_j >= N  =>  v_i.v_j <= N/2
    #
    # For vectors with entries in {-1, 0, 1} and weight w: N = w.
    # Need v_i.v_j <= w/2 (integer, so <= floor(w/2)).
    #
    # The kissing number problem in dimension 11: the known kissing number is 460.
    # But we're restricted to integer vectors. Let's find the maximum k.
    #
    # Approach: Use a greedy + local search on the conflict graph.
    # For weight w vectors, generate all C(11,w) * 2^w vectors, build conflict graph,
    # find maximum independent set.
    
    best_points = None
    best_k = 2
    
    # Start with the standard basis construction: 22 vectors (e_i and -e_i)
    # All have norm^2 = 1, and ||e_i - e_j||^2 = 2 >= 1 for i != j
    # ||e_i - (-e_j)||^2 = 2 if i != j, ||e_i - (-e_i)||^2 = 4
    basis = []
    for i in range(d):
        v1 = np.zeros(d, dtype=np.int64)
        v1[i] = 1
        v2 = np.zeros(d, dtype=np.int64)
        v2[i] = -1
        basis.append(v1)
        basis.append(v2)
    best_k = len(basis)
    best_points = np.array(basis, dtype=np.int64)
    
    # Try to add weight-2 vectors to the basis
    # Weight-2 vectors have norm^2 = 2.
    # For a weight-2 vector v to be compatible with basis vector e_i:
    #   ||v - e_i||^2 = 2 - 2*v_i + 1 >= 2 (max norm is now 2)
    #   => 3 - 2*v_i >= 2 => v_i <= 0.5 => v_i <= 0 (integer)
    #   But wait, if v_i = 1, then ||v - e_i||^2 = 2 - 2 + 1 = 1 < 2. Bad!
    #   If v_i = -1, then ||v - e_i||^2 = 2 + 2 + 1 = 5 >= 2. OK.
    #   If v_i = 0, then ||v - e_i||^2 = 2 + 0 + 1 = 3 >= 2. OK.
    # Similarly for -e_i: ||v - (-e_i)||^2 = 2 + 2*v_i + 1
    #   If v_i = 1: 2 + 2 + 1 = 5 >= 2. OK.
    #   If v_i = -1: 2 - 2 + 1 = 1 < 2. Bad!
    #   If v_i = 0: 3 >= 2. OK.
    #
    # So a weight-2 vector v is compatible with e_i iff v_i <= 0, and with -e_i iff v_i >= 0.
    # This means: for each coordinate i, if v_i = 1, it's incompatible with e_i.
    #             if v_i = -1, it's incompatible with -e_i.
    #
    # Since v has exactly 2 non-zero entries, it will be incompatible with at most 2 basis vectors.
    # But we need it compatible with ALL 22 basis vectors.
    # That's impossible if v has any non-zero entry!
    #
    # So we can't simply add weight-2 vectors to the basis set while keeping the basis.
    # We need a different approach.
    
    # Better approach: Find a large set of vectors all with the same norm^2.
    # Let's try weight-2 vectors only.
    # All weight-2 vectors have norm^2 = 2.
    # Need v.u <= 1 (since 2/2 = 1) for all pairs.
    # Two weight-2 vectors: they share 0, 1, or 2 positions.
    # If they share 0 positions: dot = 0. OK.
    # If they share 1 position: dot = ±1. OK (|dot| <= 1).
    # If they share 2 positions: dot = ±2 or 0. dot = 2 means same signs on both. Bad!
    #   dot = -2 means opposite signs on both. Bad!
    #   dot = 0 means one same, one opposite. OK.
    #
    # So the conflict condition for weight-2 vectors: they conflict iff they share
    # both positions with the same sign pattern (both + or both - on the shared positions).
    # Actually: conflict iff |dot| > 1, i.e., dot = 2 or dot = -2.
    # dot = 2: same positions, same signs.
    # dot = -2: same positions, opposite signs.
    #
    # So two weight-2 vectors conflict iff they have the same support (same 2 positions)
    # and either same or opposite sign patterns. Wait:
    # If same support {i,j}:
    #   (1,1) and (1,1): dot = 2. Conflict.
    #   (1,1) and (-1,-1): dot = -2. Conflict.
    #   (1,1) and (1,-1): dot = 0. OK.
    #   (1,1) and (-1,1): dot = 0. OK.
    #
    # So for a given support {i,j}, there are 4 sign patterns: ++, +-, -+, --.
    # Conflicts: ++ vs ++, ++ vs --, +- vs +-, +- vs --, -+ vs -+, -+ vs --, -- vs --, -- vs ++
    # Actually let's just compute: conflict iff dot > 1 or dot < -1, i.e., |dot| > 1.
    # For same support: dot is 2, 0, or -2. So conflict iff dot = ±2, i.e., same or opposite sign pattern.
    # Compatible iff dot = 0, i.e., one sign same, one different.
    #
    # So for each support {i,j}, we can pick at most 2 vectors that are compatible:
    # e.g., ++ and +- (dot = 0), or ++ and -+ (dot = 0), etc.
    # Actually: ++ is compatible with +- and -+ (dot = 0 each).
    # +- is compatible with ++ and -+ (dot = 0 each).
    # So the 4 sign patterns form a cycle of compatibility: ++ ~ +- ~ -- ~ -+ ~ ++
    # Wait: ++ and --: dot = -2. Conflict.
    # ++ and + -: dot = 0. Compatible.
    # ++ and -+: dot = 0. Compatible.
    # +- and -+: dot = -1. Compatible (|dot| = 1 <= 1).
    # +- and --: dot = -1. Compatible.
    # -+ and --: dot = 1. Compatible.
    #
    # So the conflict graph on the 4 sign patterns for a fixed support is:
    # ++ conflicts with ++ (self) and --.
    # +- conflicts with +- (self) and --.
    # -+ conflicts with -+ (self) and --.
    # -- conflicts with -- (self), ++, +-, -+.
    #
    # So -- conflicts with all others. ++ conflicts only with --.
    # Maximum independent set: {++, +-, -+} (size 3) or {+-, -+, ++} etc.
    # Actually {++, +-, -+}: ++ and +- compatible, ++ and -+ compatible, +- and -+ compatible. Yes!
    # So we can pick 3 out of 4 for each support.
    
    # Total weight-2 vectors: C(11,2) * 4 = 55 * 4 = 220.
    # Max independent set: at most 55 * 3 = 165? But there are also conflicts between
    # vectors with different supports.
    
    # Two weight-2 vectors with different supports:
    # If they share 1 position: dot = ±1. Compatible.
    # If they share 0 positions: dot = 0. Compatible.
    # So vectors with different supports are always compatible!
    
    # Therefore, the maximum independent set is: for each of the 55 supports, pick 3
    # compatible sign patterns. Total = 55 * 3 = 165.
    
    # Let's construct this explicitly.
    weight2_set = []
    for i, j in itertools.combinations(range(d), 2):
        # Pick 3 sign patterns that are mutually compatible
        # Patterns: (1,1), (1,-1), (-1,1) -- these three are mutually compatible
        v1 = np.zeros(d, dtype=np.int64)
        v1[i] = 1
        v1[j] = 1
        v2 = np.zeros(d, dtype=np.int64)
        v2[i] = 1
        v2[j] = -1
        v3 = np.zeros(d, dtype=np.int64)
        v3[i] = -1
        v3[j] = 1
        weight2_set.extend([v1, v2, v3])
    
    k_w2 = len(weight2_set)
    if k_w2 > best_k:
        best_k = k_w2
        best_points = np.array(weight2_set, dtype=np.int64)
    
    # Now try weight-3 vectors. Norm^2 = 3. Need dot <= 1 (floor(3/2) = 1).
    # Two weight-3 vectors: dot can be -3, -1, 1, 3.
    # Conflict iff |dot| > 1, i.e., dot = ±3.
    # dot = 3: same support, same signs.
    # dot = -3: same support, opposite signs.
    #
    # For a fixed support of size 3, there are 8 sign patterns.
    # Two patterns conflict iff they are identical or opposite (dot = ±3).
    # So the conflict graph is: each pattern conflicts with itself and its negative.
    # Maximum independent set: pick one from each {pattern, -pattern} pair.
    # There are 4 such pairs, so we can pick 4.
    #
    # Vectors with different supports:
    # If they share 2 positions: dot can be -2, 0, 2.
    #   |dot| = 2 > 1. Conflict!
    # If they share 1 position: dot = ±1. Compatible.
    # If they share 0 positions: dot = 0. Compatible.
    #
    # So weight-3 vectors with supports that share exactly 2 positions conflict.
    # This is more complex. Let's use greedy + local search.
    
    # Generate all weight-3 vectors
    weight3_vecs = []
    for positions in itertools.combinations(range(d), 3):
        for signs in itertools.product([-1, 1], repeat=3):
            v = np.zeros(d, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            weight3_vecs.append(v)
    
    n3 = len(weight3_vecs)  # C(11,3) * 8 = 165 * 8 = 1320
    if n3 > 0 and elapsed() < deadline - 60:
        w3_arr = np.array(weight3_vecs, dtype=np.int64)
        dots3 = w3_arr @ w3_arr.T
        conflict3 = (np.abs(dots3) > 1)
        np.fill_diagonal(conflict3, False)
        
        # Greedy with multiple restarts
        best_w3 = set()
        for attempt in range(100):
            if elapsed() > deadline - 30:
                break
            if attempt == 0:
                order = np.arange(n3)
            else:
                order = np.random.permutation(n3)
            
            selected = set()
            for idx in order:
                idx = int(idx)
                compatible = True
                for sel in selected:
                    if conflict3[idx, sel]:
                        compatible = False
                        break
                if compatible:
                    selected.add(idx)
            
            if len(selected) > len(best_w3):
                best_w3 = selected
        
        k3 = len(best_w3)
        if k3 > best_k:
            best_k = k3
            best_points = w3_arr[list(best_w3)]
    
    # Try weight-4 vectors. Norm^2 = 4. Need dot <= 2.
    # Conflict iff |dot| > 2, i.e., |dot| >= 3.
    # Two weight-4 vectors sharing 3 positions: dot can be -3, -1, 1, 3.
    #   |dot| = 3 > 2. Conflict.
    # Sharing 4 positions: dot = ±4, ±2, 0.
    #   |dot| = 4 > 2. Conflict. |dot| = 2 <= 2. OK. |dot| = 0. OK.
    
    # This gets complex. Let's try a different structural approach.
    
    # Alternative: Use the Dodecagonal lattice or other known constructions.
    # For now, let's also try mixing weights.
    
    # Let's try: all weight-2 vectors with a specific sign pattern selection,
    # plus some weight-1 vectors.
    
    # Actually, let's try to maximize using a smarter greedy.
    # Start with the 165 weight-2 vectors and try to add weight-1 vectors.
    # Weight-1 vectors: e_i and -e_i. Norm^2 = 1.
    # But max norm in the set is 2 (from weight-2 vectors).
    # So we need ||v - e_i||^2 >= 2 for all weight-2 vectors v.
    # ||v - e_i||^2 = 2 - 2*v_i + 1 = 3 - 2*v_i.
    # Need 3 - 2*v_i >= 2 => v_i <= 0.5 => v_i <= 0.
    # Similarly for -e_i: ||v + e_i||^2 = 3 + 2*v_i >= 2 => v_i >= -0.5 => v_i >= 0.
    # So v_i must be 0 for all i. But v is a weight-2 vector, so it has 2 non-zero entries.
    # Contradiction. So we can't add weight-1 vectors to the weight-2 set.
    
    # Let's try a completely different approach: use vectors with entries in {-2, -1, 0, 1, 2}.
    # This gives more flexibility.
    
    # For now, return the best we have.
    if best_points is not None and best_k >= 2:
        result = np.around(best_points).astype(np.int64)
        # Validate
        k = len(result)
        if k >= 2 and result.shape[1] == d:
            norms_sq = np.sum(result ** 2, axis=1)
            if np.all(norms_sq > 0):
                max_norm_sq = np.max(norms_sq)
                min_dist_sq = np.inf
                for i in range(k):
                    for j in range(i + 1, k):
                        diff = result[i] - result[j]
                        dist_sq = np.sum(diff ** 2)
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                if min_dist_sq >= max_norm_sq:
                    return result
    
    # Fallback
    return np.array([np.ones(d), -np.ones(d)]).astype(np.int64)

# EVOLVE-BLOCK-END
