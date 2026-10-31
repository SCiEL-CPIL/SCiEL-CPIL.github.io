"""Rewritten program for kissing number in dimension 11."""
import numpy as np
import time

D = 11
MAX_TIME = 110.0  # Safety margin before 120s kill

def construct():
    """Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm. Returns integer array of shape (k, d)."""
    start_time = time.time()
    
    # Strategy: Use the known Leech lattice kissing configuration approach.
    # For dimension 11, we can use vectors from the E8 lattice embedded appropriately,
    # or use a systematic construction based on the kissing number bounds.
    # 
    # The kissing number in dimension 11 is known to be 556.
    # We'll use a constructive approach: start with the seed program's weight-2 vectors
    # (2*C(11,2)*4 = 440 vectors) and try to add more vectors.
    #
    # Actually, let's use a more systematic approach. The known construction for
    # kissing configurations in dimension n uses:
    # 1. All weight-2 vectors: ±e_i ± e_j for i<j -> 2*C(n,2)*4 = 4n(n-1)
    #    For n=11: 4*11*10 = 440 vectors, each with norm sqrt(2)
    # 
    # But we can do better. Let's use the approach of finding a maximum set of
    # unit vectors with minimum angular separation.
    #
    # Key insight: For integer vectors, we can normalize by their norm.
    # The condition is: for all pairs (u,v), ||u-v|| >= max||w||
    # After normalizing to norm 2: ||u/||u||*2 - v/||v||*2|| >= 2
    # Which means: 2*||u/||u|| - v/||v|||| >= 2
    # ||u/||u|| - v/||v|||| >= 1
    # This means the angle between normalized vectors must be at least 60 degrees.
    #
    # For the seed: weight-2 vectors have norm sqrt(2). 
    # Distance between two such vectors: if they share no indices, ||(1,1,0...)-(1,0,1...)|| = sqrt(2)
    # If they share one index: ||(1,1,0...)-(1,-1,0...)|| = sqrt(0+4+0...) = 2
    # Max norm = sqrt(2)
    # So min distance = sqrt(2) = max norm. OK.
    #
    # Let's try a different approach: use the known optimal construction.
    # The kissing number in R^11 is 556. A construction achieving this uses
    # vectors of various weights.
    #
    # Let me try a greedy approach with random initialization and local optimization.
    
    def norm(v):
        return np.sqrt(np.sum(v**2))
    
    def valid_set(vectors):
        """Check if all pairwise distances >= max norm."""
        if len(vectors) == 0:
            return True
        norms = np.array([norm(v) for v in vectors])
        max_norm = np.max(norms)
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                d = norm(vectors[i] - vectors[j])
                if d < max_norm - 1e-9:
                    return False
        return True
    
    # Start with the seed configuration
    vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    best_vectors = vectors.copy()
    best_k = len(vectors)
    
    # Try to add more vectors using a greedy approach
    # Generate candidate vectors of weight 1, 3, 4, etc.
    # Weight 1: ±e_i -> 22 vectors, norm 1
    # But distance from e_i to e_i+e_j = 1, and max norm would be sqrt(2), so 1 < sqrt(2). Invalid.
    # 
    # Let's think about this differently. The constraint is:
    # min_{i≠j} ||v_i - v_j|| >= max_k ||v_k||
    #
    # If all vectors have the same norm R, then we need min pairwise distance >= R.
    # This is equivalent to: the normalized vectors (norm 1) have pairwise distance >= R/||v|| = 1 (if all same norm)
    # Wait: if all have norm R, then ||v_i - v_j|| >= R means ||v_i/R - v_j/R|| >= 1
    # The normalized vectors u_i = v_i/R have ||u_i - u_j|| >= 1
    # ||u_i - u_j||^2 = 2 - 2*u_i·u_j >= 1
    # u_i·u_j <= 1/2
    # So the angle between any two normalized vectors must be at least 60°.
    #
    # For the seed: all vectors have norm sqrt(2), R = sqrt(2).
    # u_i·u_j for two weight-2 vectors sharing one coordinate: 
    # e.g., (1,1,0...) and (1,-1,0...): dot = 1-1 = 0, angle = 90°. OK.
    # (1,1,0...) and (-1,1,0...): dot = -1+1 = 0. OK.
    # (1,1,0...) and (1,0,1...): dot = 1+0+0 = 1, angle = arccos(1/2) = 60°. OK (borderline).
    # So the seed is valid with exactly 60° minimum angle.
    
    # To add more vectors, we need vectors whose normalized form has dot product <= 1/2
    # with all existing normalized vectors.
    
    # Let's try adding weight-3 vectors. Norm = sqrt(3).
    # Normalized: v/sqrt(3). Dot with a weight-2 vector u/sqrt(2):
    # (v·u) / sqrt(6) <= 1/2
    # v·u <= sqrt(6)/2 ≈ 1.22
    # Since v and u are integer vectors, v·u is integer. So v·u <= 1.
    
    # For a weight-3 vector v (three ±1 entries) and weight-2 vector u (two ±1 entries):
    # v·u can range from -2 to 2 depending on overlap and signs.
    # If they share 2 coordinates: v·u = ±1 ± 1 = -2, 0, or 2
    # If they share 1 coordinate: v·u = ±1
    # If they share 0 coordinates: v·u = 0
    
    # So we need v·u <= 1 for all weight-2 vectors u.
    # If v shares 2 coordinates with u, we need the signs to give dot <= 1.
    # If v shares 1 coordinate, dot = ±1, which is OK (<=1).
    # If v shares 0 coordinates, dot = 0, OK.
    
    # The problematic case is when v shares 2 coordinates with u and the signs align to give dot = 2.
    # So we need: for every pair of coordinates in v, the corresponding weight-2 vector with matching signs should not exist in our set... but ALL weight-2 vectors exist.
    # So if v has entries at positions i,j,k with signs s_i, s_j, s_k, then the weight-2 vector with signs s_i, s_j at positions i,j will have dot product 2 with v.
    # This means NO weight-3 vector can be added! Because for any weight-3 vector, there exists a weight-2 vector in the seed that has dot product 2.
    
    # Hmm, so we can't simply add weight-3 vectors. What if we remove some weight-2 vectors to add weight-3 vectors?
    # This is getting complex. Let's try a different approach.
    
    # Alternative: Use a completely different construction.
    # The kissing number in dimension 11 is 556. 
    # One known construction uses the Leech lattice or related structures.
    
    # Let me try: use vectors of weight 2 but with a different pattern, or mix weights.
    
    # Actually, let's try a numerical optimization approach.
    # Start with the seed, then try to add random integer vectors and keep them if valid.
    
    rng = np.random.RandomState(42)
    
    def try_add(vectors, candidate):
        """Try to add candidate to vectors, return True if valid."""
        norms = np.array([np.sqrt(np.sum(v**2)) for v in vectors])
        max_norm = np.max(norms)
        cand_norm = np.sqrt(np.sum(candidate**2))
        new_max_norm = max(max_norm, cand_norm)
        
        # Check candidate against all existing
        for v in vectors:
            d = np.sqrt(np.sum((v - candidate)**2))
            if d < new_max_norm - 1e-9:
                return False
        return True
    
    # Generate candidate vectors of various weights
    candidates = []
    # Weight 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    # Weight 3
    for combo in __import__('itertools').combinations(range(D), 3):
        for signs in __import__('itertools').product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    # Weight 4
    for combo in __import__('itertools').combinations(range(D), 4):
        for signs in __import__('itertools').product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    
    # Sort candidates by norm (prefer smaller norms to keep max_norm low)
    candidates.sort(key=lambda v: np.sum(v**2))
    
    current_vectors = list(vectors)
    
    for candidate in candidates:
        if time.time() - start_time > MAX_TIME - 5:
            break
        if try_add(current_vectors, candidate):
            current_vectors.append(candidate)
    
    if len(current_vectors) > best_k:
        best_vectors = np.array(current_vectors, dtype=np.int64)
        best_k = len(current_vectors)
    
    # Try a second pass with different random ordering
    rng2 = np.random.RandomState(123)
    candidates_shuffled = list(candidates)
    rng2.shuffle(candidates_shuffled)
    
    current_vectors2 = list(vectors)
    for candidate in candidates_shuffled:
        if time.time() - start_time > MAX_TIME - 3:
            break
        if try_add(current_vectors2, candidate):
            current_vectors2.append(candidate)
    
    if len(current_vectors2) > best_k:
        best_vectors = np.array(current_vectors2, dtype=np.int64)
        best_k = len(current_vectors2)
    
    # Final verification
    if len(best_vectors) > 0:
        norms = np.array([np.sqrt(np.sum(v**2)) for v in best_vectors])
        max_norm = np.max(norms)
        valid = True
        for i in range(len(best_vectors)):
            for j in range(i+1, len(best_vectors)):
                d = np.sqrt(np.sum((best_vectors[i] - best_vectors[j])**2))
                if d < max_norm - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fall back to seed
            best_vectors = vectors
    
    return best_vectors
