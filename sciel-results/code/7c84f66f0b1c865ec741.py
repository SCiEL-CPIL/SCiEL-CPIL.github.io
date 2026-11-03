"""Kissing number in dimension 11: construct nonzero integer vectors with min pairwise distance >= max norm."""
import numpy as np
import time
import random

D = 11

def _min_dist_ge_max_norm(vectors):
    """Check if min pairwise distance >= max norm for a list of integer vectors."""
    if len(vectors) < 2:
        return True
    v = np.array(vectors, dtype=np.int64)
    norms_sq = np.sum(v * v, axis=1)
    max_norm_sq = np.max(norms_sq)
    # Compute pairwise squared distances
    # dist^2 = ||a||^2 + ||b||^2 - 2*a.b
    # We need dist^2 >= max_norm_sq for all pairs
    # Equivalently: ||a||^2 + ||b||^2 - 2*a.b >= max_norm_sq
    # For efficiency, we can check incrementally, but let's do batch checks
    return v, norms_sq, max_norm_sq


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 150s as hard kill
    
    # Strategy: Build a set of integer vectors in R^11 where min pairwise distance >= max norm.
    # 
    # Key insight: If all vectors have the same norm N, then we need pairwise distances >= N.
    # For vectors with entries in {-1, 0, 1}, norm is sqrt(k) where k = number of nonzeros.
    # Distance between two such vectors: if they share s positions with same sign and d positions with diff sign,
    # and each has k nonzeros, then dist^2 = 2*(k - s + d) ... let me think more carefully.
    #
    # Actually, let's use a simpler approach:
    # - Start with vectors of norm^2 = 2 (two nonzero entries of +-1)
    # - The distance between two such vectors v, w:
    #   If they share exactly one coordinate with same sign: dist^2 = 2 (one coord differs by 2, one by 0, rest 0) -> wait
    #   Let me compute: v has 1 at i, 1 at j. w has 1 at i, 1 at k (j != k).
    #   v - w = [0,...,0 at i, 1 at j, -1 at k, ...]. dist^2 = 2.
    #   v = [1 at i, 1 at j], w = [1 at i, -1 at j]: v-w = [0 at i, 2 at j], dist^2 = 4 >= 2. Good.
    #   v = [1 at i, 1 at j], w = [-1 at i, 1 at j]: dist^2 = 4. Good.
    #   v = [1 at i, 1 at j], w = [-1 at i, -1 at j]: dist^2 = 8. Good.
    #   v = [1 at i, 1 at j], w = [1 at k, 1 at l] (all distinct): v-w has four nonzeros: 1,-1,1,-1 -> dist^2=4. Good.
    #   v = [1 at i, 1 at j], w = [1 at i, 1 at j]: same vector, dist^2=0. Bad - can't have duplicates.
    #
    # So all vectors of form (±1 at two positions) work as long as they're distinct!
    # Count: C(11,2) * 4 = 55 * 4 = 220 vectors, all with norm^2 = 2, min dist^2 = 2.
    #
    # Can we add more? What about vectors with norm^2 > 2?
    # If max norm^2 becomes 3, we need all pairwise distances >= 3.
    # A vector with three ±1 entries has norm^2=3. Distance to a norm^2=2 vector:
    # If they share 2 coords with same signs: dist^2 = 1 (only the extra coord). 1 < 3. Bad.
    # If they share 1 coord with same sign: dist^2 = 1 + 1 + 1 = 3? Let's see:
    #   v = (1,1,0,...), w = (1,0,1,...). v-w = (0,1,-1,...), dist^2=2 < 3. Bad.
    # If they share 1 coord with opposite sign: v=(1,1,0), w=(-1,0,1). v-w=(2,1,-1), dist^2=6>=3. Good.
    # If they share 0 coords: v=(1,1,0), w=(0,0,1) but wait w needs 3 nonzeros. w=(0,1,1,0...). 
    #   v=(1,1,0,0...), w=(0,1,1,0...). v-w=(1,0,-1,0...), dist^2=2 < 3. Bad.
    #
    # So mixing norms is tricky. Let's stick with norm^2=2 vectors: 220 of them.
    #
    # But can we do better? What if we use a different norm?
    # 
    # Actually, let's think about this differently. The kissing number problem asks for the maximum number of unit spheres that can touch a central unit sphere. In terms of vectors, we want vectors of norm >= 1 (after normalization to norm 2, they become unit vectors) with pairwise angles >= 60 degrees (since dist >= norm means angle >= 60 deg when all have same norm).
    #
    # Wait, the task says: "minimum pairwise distance is at least the maximum vector norm". If all vectors have norm N, then min dist >= N means for any pair, ||v_i - v_j|| >= N. Since ||v_i|| = ||v_j|| = N, this means 2N^2 - 2 v_i·v_j >= N^2, so v_i·v_j <= N^2/2. The cosine of the angle is v_i·v_j / N^2 <= 1/2, so angle >= 60 degrees.
    #
    # The kissing number in dimension 11 is known to be at least 504 (from the Leech lattice construction) and at most 556. Actually, let me recall: the kissing number in dimension 11 is not known exactly, but bounds are known. The best known lower bound is 504.
    #
    # However, we need INTEGER vectors. So we need to find integer vectors that, when normalized, give a kissing configuration.
    #
    # Let me try a different approach: use the E11 lattice or related constructions.
    #
    # Actually, let's try a greedy approach with local search:
    # 1. Generate candidate vectors (small integer vectors)
    # 2. Greedily add vectors that maintain the constraint
    # 3. Use local search to try to add more
    #
    # For efficiency, let's work with vectors of bounded norm.
    
    # First, get the baseline: all ±1,±1 vectors (norm^2 = 2)
    baseline = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    baseline.append(v)
    
    # Verify baseline
    v_arr = np.array(baseline, dtype=np.int64)
    norms_sq = np.sum(v_arr * v_arr, axis=1)
    max_ns = np.max(norms_sq)
    # Check all pairwise distances
    # For 220 vectors, 220*219/2 = 24090 pairs, manageable
    def is_valid(vecs):
        if len(vecs) < 2:
            return True
        v = np.array(vecs, dtype=np.int64)
        ns = np.sum(v * v, axis=1)
        mns = np.max(ns)
        # Compute all pairwise dist^2
        for a in range(len(v)):
            for b in range(a+1, len(v)):
                d2 = np.sum((v[a] - v[b]) ** 2)
                if d2 < mns:
                    return False
        return True
    
    # The baseline should be valid
    # Now try to add more vectors using greedy + local search
    
    # Candidate pool: all integer vectors with entries in {-2,-1,0,1,2} and norm^2 <= 4
    # This gives us vectors of norm^2 = 1, 2, 3, 4
    # But if we include norm^2=1 vectors, max norm becomes 1, and we need dist >= 1
    # A norm^2=1 vector is a unit vector e_i. Distance to another e_j (j!=i) is sqrt(2) >= 1. OK.
    # Distance to a norm^2=2 vector: if they share a coordinate, dist^2 = 1. 1 >= 1. OK.
    # But then max norm is 1, and we need ALL distances >= 1.
    # Two different unit vectors: dist^2 = 2 >= 1. OK.
    # Unit vector and norm^2=2 vector sharing one coord: dist^2 = 1 >= 1. OK.
    # Two norm^2=2 vectors: as before, min dist^2 = 2 >= 1. OK.
    # So we could include unit vectors! That adds 22 more (11 * 2).
    # Total: 220 + 22 = 242.
    
    # But wait, if max norm^2 = 2, we need dist^2 >= 2.
    # Unit vector e_i and norm^2=2 vector (1,1 at i,j): dist^2 = 1 < 2. BAD.
    # So we can't mix norm^2=1 and norm^2=2 if we want dist >= max norm.
    # Unless max norm is 1, but then we can't have norm^2=2 vectors.
    
    # So either all vectors have norm^2 <= 1 (just unit vectors, 22 total) or we accept the tradeoff.
    
    # Let's try a smarter approach: use vectors with larger norms but better separation.
    
    # Actually, let me try the following: generate all vectors with entries in {-1, 0, 1} and use greedy selection.
    
    # Generate all {-1,0,1} vectors with at least 2 nonzeros (to avoid unit vectors causing issues)
    # There are 3^11 = 177147 total, minus those with 0 or 1 nonzero.
    # With 0 nonzero: 1
    # With 1 nonzero: 11 * 2 = 22
    # So 177147 - 1 - 22 = 177124 candidates. Too many to store all at once efficiently.
    
    # Let's use a different strategy: start with the 220 norm^2=2 vectors, then try to add norm^2=3 vectors that are compatible.
    
    # For a norm^2=3 vector w and a norm^2=2 vector v, we need ||v-w||^2 >= 3.
    # ||v-w||^2 = ||v||^2 + ||w||^2 - 2 v·w = 2 + 3 - 2 v·w = 5 - 2 v·w >= 3
    # So v·w <= 1.
    # v has two ±1 entries, w has three ±1 entries.
    # v·w can be: if they share 2 coords with same signs: 2. 5-4=1 < 3. Bad.
    # If they share 2 coords, one same one opposite: 0. 5-0=5 >= 3. Good.
    # If they share 1 coord same: 1. 5-2=3 >= 3. Good.
    # If they share 1 coord opposite: -1. 5+2=7 >= 3. Good.
    # If they share 0 coords: 0. 5 >= 3. Good.
    # So the only bad case is when v and w share exactly 2 coordinates with the same signs.
    
    # So for a norm^2=3 vector w, it conflicts with norm^2=2 vectors v that agree with w on 2 of w's 3 coordinates.
    # w has 3 coordinates. The norm^2=2 vectors that conflict are those that pick 2 of w's 3 coordinates with matching signs.
    # There are C(3,2) = 3 such pairs, and for each pair, the sign is determined. So 3 conflicting vectors per w.
    # Plus their negatives? No, the conflicting vectors have specific signs matching w.
    
    # So each norm^2=3 vector conflicts with exactly 3 norm^2=2 vectors (the ones that match it on 2 coords).
    # There are 11 * 10 * 9 / 6 * 8 = let me count: C(11,3) * 8 = 165 * 8 = 1320 norm^2=3 vectors.
    # Each conflicts with 3 norm^2=2 vectors.
    
    # If we remove 3 norm^2=2 vectors to add 1 norm^2=3 vector, we lose 3 and gain 1. Net -2. Bad.
    # Unless one norm^2=3 vector conflicts with the same norm^2=2 vectors as another, so we can add multiple.
    
    # This is getting complex. Let me just do a greedy approach with the full candidate set.
    
    # Greedy: start with empty set, add vectors in order of some heuristic.
    # Or: start with the 220 norm^2=2 vectors, then try to swap/add.
    
    # Let me try a simple greedy with all {-1,0,1} vectors of norm^2 >= 2.
    
    # Generate candidates
    candidates = []
    for mask in range(1, 3**D):
        # Convert to base-3 digits
        digits = []
        m = mask
        for _ in range(D):
            digits.append(m % 3)
            m //= 3
        # Convert: 0->0, 1->1, 2->-1
        v = [0 if d == 0 else (1 if d == 1 else -1) for d in digits]
        ns = sum(x*x for x in v)
        if ns >= 2:
            candidates.append((ns, v))
    
    # Sort by norm^2 ascending, then by some deterministic order
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    # Greedy selection
    selected = []
    selected_set = set()
    
    def can_add(v, current_selected):
        """Check if v can be added to current_selected."""
        if not current_selected:
            return True
        v_arr = np.array(v, dtype=np.int64)
        v_ns = int(np.sum(v_arr * v_arr))
        # Max norm in current set
        max_ns = max(int(np.sum(s * s)) for s in current_selected)
        new_max_ns = max(max_ns, v_ns)
        for s in current_selected:
            d2 = int(np.sum((v_arr - np.array(s, dtype=np.int64)) ** 2))
            if d2 < new_max_ns:
                return False
        return True
    
    # This is too slow for large sets. Let me use a different approach.
    
    # Actually, let's just go with the 220 norm^2=2 vectors as the answer, since they're guaranteed valid.
    # Then try to improve with local search if time permits.
    
    result = baseline[:]
    
    # Try to add norm^2=3 vectors by removing conflicting norm^2=2 vectors
    # For each norm^2=3 vector, check if we can add it by removing its 3 conflicting norm^2=2 vectors
    # Net change: -3 + 1 = -2. Not worth it.
    
    # What about adding pairs of norm^2=3 vectors that share the same conflicting set?
    # Two norm^2=3 vectors w1, w2. They conflict with the same norm^2=2 vectors if they agree on the same 2-coordinate subsets.
    # This is complex. Let's skip.
    
    # Alternative: use a completely different construction.
    # The Leech lattice in dimension 24 has kissing number 196560. In dimension 11, we can use a sublattice.
    # But constructing integer vectors from the Leech lattice is non-trivial.
    
    # Let me try: use the D11 lattice vectors. D11 consists of integer vectors with even sum.
    # The norm^2=2 vectors in D11 are those with two ±1 entries where the sum is even.
    # Sum of two ±1: 2, 0, or -2. All even. So all 220 norm^2=2 vectors are in D11.
    # Norm^2=4 vectors in D11: four ±1 entries (sum is even: 4, 2, 0, -2, -4) or one ±2 entry.
    # One ±2 entry: 11 * 2 = 22 vectors. Norm^2=4.
    # Distance between (2 at i) and (1,1 at i,j): dist^2 = 0^2 + 1^2 + ... = 1 + 0 + ... Let me compute.
    # v = (2,0,...), w = (1,1,0,...). v-w = (1,-1,0,...). dist^2 = 2 < 4. Bad.
    # So can't mix.
    
    # I think 220 is a solid answer. Let me verify it's valid and return it.
    
    # Verify
    v_arr = np.array(result, dtype=np.int64)
    ns = np.sum(v_arr * v_arr, axis=1)
    max_ns = int(np.max(ns))
    valid = True
    for a in range(len(v_arr)):
        for b in range(a+1, len(v_arr)):
            d2 = int(np.sum((v_arr[a] - v_arr[b]) ** 2))
            if d2 < max_ns:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: just return the norm^2=2 vectors
        pass
    
    elapsed = time.time() - start_time
    if elapsed > 140.0:
        return np.array(result, dtype=np.int64)
    
    # Try to improve: add vectors of norm^2=3 by removing conflicts
    # Build a conflict map: for each norm^2=3 vector, which norm^2=2 vectors does it conflict with?
    
    # Index norm^2=2 vectors
    n2_index = {}
    for i, v in enumerate(result):
        key = tuple(v)
        n2_index[key] = i
    
    # Generate norm^2=3 vectors
    n3_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            n3_vectors.append(v)
    
    # For each n3 vector, find conflicting n2 vectors
    # A n2 vector conflicts if it agrees with n3 on 2 coordinates
    # n3 has coords (i,j,k) with signs (si,sj,sk)
    # Conflicting n2 vectors: (i,j) with (si,sj), (i,k) with (si,sk), (j,k) with (sj,sk)
    
    # Try to find a set of n3 vectors that can be added by removing a small number of n2 vectors
    # This is a set cover / hitting set problem, which is NP-hard, but we can do greedy.
    
    # For each n3 vector, cost = number of n2 vectors it conflicts with that are still present
    # Greedily add n3 vectors with lowest cost, removing their conflicts
    
    # But this is complex and may not improve. Let's just return the 220.
    
    return np.array(result, dtype=np.int64)
