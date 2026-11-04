import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11

    # Type A vectors: ±2e_i for i in 0..10, giving 22 vectors
    type_a = []
    for i in range(D):
        for sign in (-2, 2):
            v = np.zeros(D, dtype=np.int8)
            v[i] = sign
            type_a.append(v)
    type_a = np.array(type_a, dtype=np.int8)

    # Type D vectors: vectors with exactly 4 nonzero coordinates, each ±1
    # These have norm 2 (squared norm = 4)
    # Two Type D vectors are compatible iff their dot product has absolute value <= 2
    # (since min distance >= max norm means ||v_i - v_j||^2 >= max(||v_i||^2, ||v_j||^2) = 4)
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j = 8 - 2*v_i.v_j >= 4
    # So v_i.v_j <= 2

    # Generate all 4-subsets
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330

    # For each support, generate all 16 sign patterns
    # But we can exploit symmetry: for a given support, we only need one representative
    # per equivalence class under sign flips of individual coordinates and permutation of support elements.
    # Actually, for a fixed support S of size 4, the 16 sign patterns form an antipodal set.
    # Two vectors with the same support have dot product = sum of 4 products of signs = ±4, ±2, ±0
    # Wait: if same support, dot = sum_{i in S} s_i * t_i where s_i, t_i ∈ {±1}
    # This equals 4 - 2*(number of positions where they differ)
    # So dot can be 4, 2, 0, -2, -4
    # We need dot <= 2, so we cannot have two vectors with the same support and dot = 4 (same sign pattern)
    # Actually we CAN have them if dot <= 2. dot=4 is forbidden.
    # So for a given support, we can pick at most 15 of the 16 sign patterns (all except one).
    # But actually, we want to maximize the total count, so we should pick as many as possible.

    # Let's use a different approach: model as a graph where nodes are all candidate vectors
    # and edges connect incompatible pairs, then find maximum independent set.
    # But that's too large. Instead, let's be smart about it.

    # Key insight: For Type D vectors, two vectors with supports S and T are compatible
    # iff |dot(v_S, v_T)| <= 2 where the dot depends on the sign patterns.
    # If |S ∩ T| <= 1, then dot = 0 regardless of signs, so always compatible.
    # If |S ∩ T| = 2, then dot = sum of 2 sign products = ±2 or 0. Always <= 2. Compatible.
    # If |S ∩ T| = 3, then dot = sum of 3 sign products = ±3 or ±1. Need |dot| <= 2, so dot = ±1 is ok, ±3 is not.
    # If |S ∩ T| = 4 (same support), dot = sum of 4 sign products = ±4, ±2, 0. Need dot <= 2, so -4, -2, 0, 2 are ok, but 4 is not.

    # So the constraint is:
    # - Same support: cannot have two vectors with identical sign patterns (dot=4)
    # - Support intersection of size 3: cannot have sign patterns where the 3 shared coordinates all agree (dot contribution from shared = 3, plus the 4th coordinate contributes ±1, total = ±4 or ±2). Wait let me recompute.

    # If S and T share 3 elements, say S = {a,b,c,d}, T = {a,b,c,e}.
    # dot = s_a*t_a + s_b*t_b + s_c*t_c + s_d*0 + 0*t_e = s_a*t_a + s_b*t_b + s_c*t_c
    # This is sum of 3 terms each ±1, so dot ∈ {-3,-1,1,3}
    # Need dot <= 2, so dot ∈ {-3,-1,1} are ok, dot = 3 is not.
    # dot = 3 means all three shared coordinates have same sign in both vectors.

    # Strategy: For each support, we can include at most one vector from each "sign class".
    # Actually, let's just do a greedy approach on the full set of Type D vectors,
    # but use the structure to speed it up.

    # Total Type D vectors: 330 supports * 16 signs = 5280
    # That's manageable for a greedy with some structure.

    # Let's precompute all Type D vectors
    all_type_d = []
    sup_to_vectors = defaultdict(list)
    for s in supports:
        vecs = []
        for bits in range(16):
            v = np.zeros(D, dtype=np.int8)
            for b, idx in enumerate(s):
                v[idx] = 1 if (bits >> b) & 1 else -1
            vecs.append(v)
        sup_to_vectors[s] = vecs
        all_type_d.extend(vecs)

    all_type_d = np.array(all_type_d, dtype=np.int8)
    n_td = len(all_type_d)  # 5280

    # For efficiency, let's work with support indices
    # Two Type D vectors with supports s1 (index i) and s2 (index j) and sign patterns p1, p2
    # are compatible iff:
    #   if i == j: p1 != p2 (i.e., not identical sign pattern)
    #   if |s1 ∩ s2| == 3: not all three shared coordinates have same sign
    #   otherwise: always compatible

    # Let's precompute for each pair of supports whether they conflict (intersection size 3)
    # and which sign patterns conflict.

    # For supports with intersection size 3, the conflicting sign patterns are those where
    # the 3 shared coordinates all agree in sign.

    # Let's build the conflict structure more carefully.
    # For each pair (i, j) of supports with |S_i ∩ S_j| = 3:
    #   Let shared = S_i ∩ S_j (size 3)
    #   Two sign patterns p, q conflict if for all k in shared, sign_p(k) == sign_q(k)
    #   There are 2^3 = 8 such pairs of sign patterns that conflict (for each assignment of signs to the 3 shared coords, both p and q must match)

    # This is getting complex. Let me just do a vectorized check on the full set.
    # 5280 vectors, pairwise check is 5280^2/2 ≈ 14M pairs. With NumPy, this might be fast enough.

    # Actually, let's try a simpler greedy: iterate through all Type D vectors in order,
    # and add a vector if it's compatible with all previously added ones.

    # To make it faster, let's use the support structure.
    # For a new vector with support S and sign pattern P:
    #   - It conflicts with any already-added vector with the same support and same sign pattern.
    #   - It conflicts with any already-added vector with support T where |S∩T|=3 and the 3 shared coords all agree.

    # Let's maintain for each support, which sign patterns have been used.
    used_signs = defaultdict(set)  # support -> set of sign pattern indices used

    # For the intersection-of-3 constraint, we need to track more.
    # For each triple of coordinates, track which sign patterns on that triple have been "blocked".
    # A vector with support S and sign pattern P blocks, for any other support T sharing 3 coords with S,
    # the sign patterns Q on T that agree with P on the shared 3 coords.

    # Let's define: for each triple t and each sign pattern on t (8 possibilities),
    # track how many vectors have been added that "cover" this (t, pattern) pair.
    # A new vector is valid if for its support S and sign pattern P:
    #   - (S, P) is not already used
    #   - For each triple t ⊂ S, the (t, P|_t) is not "full" in a way that blocks

    # Actually, the constraint is pairwise, not global. Let me just do the direct check.

    # Given 5280 vectors and the need to check compatibility with all previously added,
    # and assuming we add ~3000 of them, that's ~3000*5280/2 ≈ 8M checks.
    # Each check is O(1) if we precompute dot products... but that's 5280*5280/2 ≈ 14M dots.
    # With vectorized NumPy, computing all pairwise dots is 5280^2 = 28M operations, feasible.

    # Let's precompute the Gram matrix of Type D vectors.
    # G[i,j] = all_type_d[i] . all_type_d[j]
    # A pair (i,j) is compatible iff G[i,j] <= 2

    # 5280x5280 matrix of int8 dots: each dot is in [-4,4], fits in int8.
    # Memory: 5280*5280 bytes ≈ 28MB. Fine.

    if time.time() > deadline - 100.0:
        # Not enough time, just return Type A
        return type_a

    # Compute Gram matrix
    G = all_type_d.astype(np.int32) @ all_type_d.T

    # Greedy: iterate through vectors, add if compatible with all already added
    # To get a good packing, we can try multiple orderings or use a smarter heuristic.
    # Let's try: for each vector, count how many already-added vectors it conflicts with.
    # Pick the one with 0 conflicts and highest "degree" (most conflicts with remaining).

    # Simpler: just iterate in a fixed order and greedily add.
    # But to get a better result, let's randomize the order a few times.

    best_selected = None
    best_count = 0

    n_trials = 10
    for trial in range(n_trials):
        if time.time() > deadline - 20.0:
            break

        if trial == 0:
            order = np.arange(n_td)
        else:
            order = np.random.permutation(n_td)

        selected = []
        added = np.zeros(n_td, dtype=bool)

        for idx in order:
            if time.time() > deadline - 15.0:
                break
            # Check if idx is compatible with all already added
            if added[idx]:
                continue
            # Get indices of already added vectors
            added_idx = np.where(added)[0]
            if len(added_idx) == 0:
                added[idx] = True
                selected.append(idx)
                continue
            # Check dot products
            dots = G[idx, added_idx]
            if np.all(dots <= 2):
                added[idx] = True
                selected.append(idx)

        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected[:]

    # Construct final vector set
    type_d_selected = all_type_d[best_selected] if best_selected else np.array([], dtype=np.int8).reshape(0, D)

    # Combine Type A and selected Type D
    all_vecs = np.vstack([type_a, type_d_selected])

    # Verify
    n = len(all_vecs)
    if n > 1:
        norms_sq = np.sum(all_vecs**2, axis=1)
        max_norm_sq = int(np.max(norms_sq))
        # Check pairwise distances: ||v_i - v_j||^2 >= max_norm_sq
        # For Type A: norm^2 = 4
        # For Type D: norm^2 = 4
        # So max_norm_sq should be 4
        # Need ||v_i - v_j||^2 >= 4 for all pairs

        # Type A - Type A: ||±2e_i - ±2e_j||^2 = 8 if i≠j, 0 if same (but we have distinct)
        # Actually we have both +2e_i and -2e_i, so ||2e_i - (-2e_i)||^2 = 16, ||2e_i - 2e_j||^2 = 8
        # Type A - Type D: ||±2e_i - v_D||^2 = 4 + 4 - 2*(±2*v_D[i]) = 8 ∓ 4*v_D[i]
        # If v_D[i] = ±1, this is 8 ∓ 4*(±1) = 8-4=4 or 8+4=12. Min is 4. OK.
        # Type D - Type D: checked by Gram matrix, ||v_i - v_j||^2 = 8 - 2*G[i,j] >= 4 iff G[i,j] <= 2. OK.

        # Let's do a spot check for safety
        ok = True
        if n <= 10000:
            # Full verification
            for i in range(0, n, 100):
                if time.time() > deadline - 5.0:
                    break
                batch = all_vecs[i:i+100]
                for j in range(i + 100, n, 100):
                    if time.time() > deadline - 5.0:
                        ok = False
                        break
                    batch2 = all_vecs[j:j+100]
                    # Compute pairwise squared distances
                    d2 = np.sum(batch**2, axis=1, keepdims=True) + np.sum(batch2**2, axis=1).T - 2 * batch.astype(np.int32) @ batch2.T
                    if np.min(d2) < max_norm_sq:
                        ok = False
                        break
                if not ok:
                    break

        if not ok:
            # Fallback: just Type A
            return type_a

    return all_vecs
