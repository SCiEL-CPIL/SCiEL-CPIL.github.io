"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin
    
    # Start with the known valid set: all ±(e_i + e_j) for i < j
    vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    k = len(vectors)
    
    # Try to add more vectors from a larger candidate pool
    # Candidate pool: vectors with entries in {-1, 0, 1}
    # Generate candidates with 1 nonzero entry (unit vectors ±e_i)
    # These have norm 1, but our current max norm is sqrt(2)
    # Distance from ±e_i to any (e_a + e_b): 
    #   If i not in {a,b}: dist² = 1 + 1 + 1 = 3 (e_i has 1 at i, target has 1 at a,b)
    #   Wait: e_i - (e_a+e_b): if i≠a,b, then at i: 1-0=1, at a: 0-1=-1, at b: 0-1=-1. dist²=3.
    #   If i=a: e_a - (e_a+e_b) = -e_b. dist²=1. But max norm is sqrt(2), so 1 < sqrt(2). FAILS.
    # So unit vectors don't work with the current set.
    
    # What about vectors with 3 nonzero entries?
    # (e_i + e_j + e_k): norm = sqrt(3)
    # Distance to (e_a + e_b): depends on overlap.
    # If {i,j,k} ∩ {a,b} = ∅: dist² = 3 + 2 = 5. dist = sqrt(5). max norm = sqrt(3). sqrt(5) >= sqrt(3) ✓
    # If |intersection| = 1: say i=a. (e_i+e_j+e_k) - (e_i+e_b) = e_j + e_k - e_b. 
    #   If b not in {i,j,k}: dist² = 1+1+1 = 3. max norm = sqrt(3). 3 >= 3 ✓ (dist=sqrt(3))
    #   If b in {j,k}: say b=j. (e_i+e_j+e_k)-(e_i+e_j) = e_k. dist²=1. max norm=sqrt(3). 1 < 3. FAILS.
    # So (e_i+e_j+e_k) fails if it shares 2 positions with (e_i+e_j) type vectors.
    
    # This is getting complex. Let me try a different approach: use a greedy algorithm
    # with a larger candidate pool and check validity efficiently.
    
    # Generate a large candidate pool
    candidates = []
    # All vectors with exactly 2 nonzero entries (already in our set)
    # Let's try vectors with 3 nonzero entries
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
                            candidates.append(v)
    
    # For each candidate, check if it can be added
    def can_add(vec, vecs):
        """Check if vec can be added to vecs maintaining the constraint."""
        vec = np.array(vec, dtype=np.int64)
        vec_norm_sq = int(np.dot(vec, vec))
        
        # New max norm squared = max(current max, vec_norm_sq)
        # Current max norm sq is sqrt(2)^2 = 2 for our initial set
        # But as we add vectors, it changes.
        # Let's compute current max norm sq
        current_norms_sq = np.sum(vecs ** 2, axis=1)
        current_max_norm_sq = int(np.max(current_norms_sq))
        new_max_norm_sq = max(current_max_norm_sq, vec_norm_sq)
        
        # Check all pairwise distances with existing vectors
        diffs = vecs - vec  # shape (n, D)
        dists_sq = np.sum(diffs ** 2, axis=1)
        if np.min(dists_sq) < new_max_norm_sq - 1e-9:
            return False
        return True
    
    added_count = 0
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        if can_add(cand, vectors):
            vectors = np.vstack([vectors, np.array(cand, dtype=np.int64)])
            added_count += 1
    
    # Try local search: swap operations
    # For time, let's also try adding vectors with 4 nonzero entries if there's time
    if time.time() < deadline - 30:
        candidates4 = []
        for i in range(D):
            for j in range(i+1, D):
                for k in range(j+1, D):
                    for l in range(k+1, D):
                        for si in (1, -1):
                            for sj in (1, -1):
                                for sk in (1, -1):
                                    for sl in (1, -1):
                                        v = [0] * D
                                        v[i] = si
                                        v[j] = sj
                                        v[k] = sk
                                        v[l] = sl
                                        candidates4.append(v)
        
        random.shuffle(candidates4)
        for cand in candidates4:
            if time.time() > deadline - 10:
                break
            if can_add(cand, vectors):
                vectors = np.vstack([vectors, np.array(cand, dtype=np.int64)])
    
    return vectors

# EVOLVE-BLOCK-END
