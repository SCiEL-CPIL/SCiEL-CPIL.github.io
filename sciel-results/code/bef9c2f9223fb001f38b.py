"""Kissing number in dimension 11: maximize number of integer vectors with min pairwise distance >= max norm."""
import time
import numpy as np
from itertools import combinations

D = 11

def _make_e8_set():
    """Generate 240 minimal vectors of E8 lattice (squared norm 4, all coordinates 0 or +/-1)."""
    vecs = []
    # Type 1: exactly two coordinates are +/-1, rest 0. Choose 2 from 8, choose signs (4 combos)
    for i, j in combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 8
                v[i] = si
                v[j] = sj
                vecs.append(tuple(v))
    # Type 2: all 8 coordinates are +/-1, even number of -1s
    for mask in range(256):
        bits = bin(mask).count('1')
        if bits % 2 == 0:
            v = tuple(1 - 2 * int(b) for b in bin(mask)[2:].zfill(8))
            # v has +1 where bit is 0, -1 where bit is 1
            # Check: -1 count = bits, which is even
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def _try_add_vector(v, existing, norms_sq, max_norm_sq):
    """Try to add vector v to existing set. Return True if valid."""
    # Check if v is nonzero
    if np.all(v == 0):
        return False
    v_norm_sq = int(np.sum(v * v))
    if v_norm_sq == 0:
        return False
    # Check distance to all existing vectors
    # dist_sq = ||v||^2 + ||u||^2 - 2*v.u
    dots = existing @ v
    dist_sq = v_norm_sq + norms_sq - 2 * dots
    min_dist_sq = np.min(dist_sq)
    if min_dist_sq < max_norm_sq - 1e-9:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Build E8 in first 8 dimensions, extended with 3 zeros
    e8_8d = _make_e8_set()  # 240 vectors, shape (240, 8)
    e8_11d = np.hstack([e8_8d, np.zeros((len(e8_8d), 3), dtype=np.int64)])
    
    # Also consider the dual-like set: E8 shifted or scaled? 
    # Actually, let's also consider adding the "opposite" set or a complementary structure.
    # The kissing number of E8 is 240. In R^11, we can potentially add more.
    
    # Let's start with E8 x 1 (i.e., E8 in first 8 dims, 0 in last 3)
    # And also consider the "weight-2" vectors in the last 3 dims combined with E8?
    
    # Actually, a better approach: use the known kissing configuration for R^11.
    # The kissing number of R^11 is 556 (conjectured/known).
    # But we need INTEGER vectors with the specific distance condition.
    
    # Let's try a greedy approach starting from a good seed.
    
    # Candidate pool: 
    # 1. E8 in first 8 dims (240 vectors)
    # 2. Weight-2 vectors in all 11 dims (55*4 = 220 vectors)
    # 3. Weight-1 vectors (11*2 = 22 vectors) - but these have norm 1, while E8 has norm 2
    #    The condition is: min pairwise distance >= max norm
    #    If we mix norm 1 and norm 2 vectors, max norm is 2.
    #    Distance between norm 1 and norm 2 vectors: need >= 2.
    #    This is restrictive.
    
    # Let's focus on vectors of the same norm for simplicity.
    # E8 vectors have squared norm 4 (norm 2).
    # Weight-2 vectors have squared norm 2 (norm sqrt(2)).
    
    # If we use only norm-2 vectors (squared norm 4), we need min distance >= 2.
    # dist_sq >= 4.
    # ||v||^2 + ||u||^2 - 2 v.u >= 4
    # 4 + 4 - 2 v.u >= 4
    # v.u <= 2
    
    # Let's build a pool of integer vectors with squared norm 4.
    # Types:
    # A: Two coords are +/-1, rest 0. (norm^2 = 2) -- wait, 1^2 + 1^2 = 2. Not 4.
    # B: One coord is +/-2, rest 0. (norm^2 = 4). 11*2 = 22 vectors.
    # C: Two coords are +/-1, one coord is +/-1, rest 0? No, 1+1+1=3.
    # D: Four coords are +/-1, rest 0. (norm^2 = 4). C(11,4)*16 = 330*16 = 5280 vectors.
    # E: Two coords are +/-1, and one coord is... no.
    
    # So for squared norm 4:
    # Type B: 22 vectors (one coord is +/-2)
    # Type D: 5280 vectors (four coords are +/-1)
    
    # Total pool: 22 + 5280 = 5302 vectors.
    
    # Let's generate this pool and do greedy + local search.
    
    pool = []
    
    # Type B: one coord is +/-2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            pool.append(v)
    
    # Type D: four coords are +/-1
    for idx in combinations(range(D), 4):
        for signs in range(16):
            v = np.zeros(D, dtype=np.int64)
            for k, i in enumerate(idx):
                if (signs >> k) & 1:
                    v[i] = -1
                else:
                    v[i] = 1
            pool.append(v)
    
    pool = np.array(pool, dtype=np.int64)
    n_pool = len(pool)
    
    # Greedy: start with a random vector, then add vectors that satisfy the condition.
    # Condition: for all u in current set, dist_sq(v, u) >= max_norm_sq
    # Since all have norm^2 = 4, max_norm_sq = 4.
    # dist_sq = 4 + 4 - 2*v.u = 8 - 2*v.u >= 4 => v.u <= 2
    
    # Let's use a simple greedy with random restarts, plus a Kempe-chain swap improvement.
    
    best_set = None
    best_size = 0
    
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Random initial vector
        start_idx = np.random.randint(n_pool)
        current = [pool[start_idx]]
        used = [start_idx]
        used_set = set([start_idx])
        
        # Greedy add
        remaining = [i for i in range(n_pool) if i != start_idx]
        np.random.shuffle(remaining)
        
        for idx in remaining:
            if time.time() > deadline - 20:
                break
            v = pool[idx]
            valid = True
            for u_idx in used:
                u = pool[u_idx]
                dot = int(np.dot(v, u))
                if dot > 2:
                    valid = False
                    break
            if valid:
                current.append(idx)
                used.append(idx)
                used_set.add(idx)
        
        current_set = np.array([pool[i] for i in current])
        size = len(current)
        
        if size > best_size:
            best_size = size
            best_set = current_set.copy()
        
        # Local search: Kempe-chain swaps
        # Try to remove one vector and add one or more vectors.
        improved = True
        n_iterations = 0
        max_iterations = 500
        while improved and n_iterations < max_iterations:
            improved = False
            n_iterations += 1
            if time.time() > deadline - 10:
                break
            
            # Try removing each vector in the current set
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                # Remove current[i]
                removed_idx = current[i]
                new_current = current[:i] + current[i+1:]
                
                # Try to add any vector from the pool not in new_current
                added = False
                for j in range(n_pool):
                    if j in used_set and j != removed_idx:
                        continue
                    if j in new_current:
                        continue
                    v = pool[j]
                    valid = True
                    for u_idx in new_current:
                        u = pool[u_idx]
                        dot = int(np.dot(v, u))
                        if dot > 2:
                            valid = False
                            break
                    if valid:
                        new_current.append(j)
                        added = True
                        break
                if added:
                    current = new_current
                    used_set.remove(removed_idx)
                    # Update used_set for the new addition
                    # We need to track this properly
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = np.array([pool[i] for i in current]).copy()
                    break  # Restart the outer loop
        
        # Note: The local search above is simplified. A proper Kempe-chain would be better.
    
    if best_set is None:
        # Fallback to the seed solution
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        vectors.append(v)
        best_set = np.array(vectors, dtype=np.int64)
    
    return best_set
