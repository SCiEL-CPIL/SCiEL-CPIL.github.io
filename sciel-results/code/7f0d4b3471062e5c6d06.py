import itertools
import time
import random
import numpy as np

N = 16
V = N * N
ALL_MASK = (1 << V) - 1

def _precompute():
    """Precompute distance matrix and conflict masks."""
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Compute squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2)
    
    # Precompute conflict masks
    # C[i][j] is the mask of points k that form an isosceles triangle with i and j
    # i.e., d(i,j)==d(i,k) or d(i,j)==d(j,k) or d(i,k)==d(j,k)
    
    # To speed this up, we can precompute for each point i, a dict of distance -> mask of points
    dist_masks = [{} for _ in range(V)]
    for i in range(V):
        for j in range(V):
            if i == j: continue
            d = int(D[i, j])
            if d not in dist_masks[i]:
                dist_masks[i][d] = 0
            dist_masks[i][d] |= (1 << j)
            
    # Precompute bisector masks: B[i][j] is mask of k such that d(i,k)==d(j,k)
    # This is the perpendicular bisector of i and j (in terms of squared distance equality)
    # Note: This is not a geometric line, just the set of points equidistant from i and j.
    bisector_masks = [[0] * V for _ in range(V)]
    for i in range(V):
        for j in range(i + 1, V):
            mask = 0
            # Optimization: Instead of iterating all k, we can intersect sets of distances?
            # d(i,k) = d(j,k)
            # We can just iterate k. V=256, so 256*256*256 is ~16M ops. Acceptable.
            for k in range(V):
                if k != i and k != j:
                    if D[i, k] == D[j, k]:
                        mask |= (1 << k)
            bisector_masks[i][j] = mask
            bisector_masks[j][i] = mask

    C = [[0] * V for _ in range(V)]
    for i in range(V):
        for j in range(i + 1, V):
            dij = int(D[i, j])
            mask = 0
            # k where d(i,k) == d(i,j)
            mask |= dist_masks[i].get(dij, 0)
            # k where d(j,k) == d(i,j)
            mask |= dist_masks[j].get(dij, 0)
            # k where d(i,k) == d(j,k)
            mask |= bisector_masks[i][j]
            # Remove i and j
            mask &= ~(1 << i)
            mask &= ~(1 << j)
            C[i][j] = mask
            C[j][i] = mask
            
    return points, C

_points, _C = None, None

def _init():
    global _points, _C
    if _points is None:
        _points, _C = _precompute()

def _get_blocked(S_mask):
    """Calculate the union of conflict masks for all pairs in S."""
    blocked = 0
    indices = []
    m = S_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        indices.append(idx)
        m ^= lsb
        
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            blocked |= _C[indices[i]][indices[j]]
            
    return blocked

def _is_maximal(S_mask):
    """Check if S is maximal."""
    if S_mask == 0:
        return ALL_MASK == 0 # False unless V=0
    blocked = _get_blocked(S_mask)
    return (blocked | S_mask) == ALL_MASK

def _is_valid(S_mask):
    """Check if S is a valid no-isosceles set."""
    indices = []
    m = S_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        indices.append(idx)
        m ^= lsb
        
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            if S_mask & _C[indices[i]][indices[j]]:
                return False
    return True

def _greedy_build(seed):
    """Greedy construction."""
    rng = random.Random(seed)
    S_mask = 0
    blocked = 0
    
    while True:
        candidates_mask = ALL_MASK & ~blocked & ~S_mask
        
        if not candidates_mask:
            break
            
        candidates = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            m ^= lsb
            
        if not candidates:
            break
            
        # Score candidates
        curr_indices = []
        m = S_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            curr_indices.append(idx)
            m ^= lsb
            
        best_score = -1
        best_candidate = None
        best_new_blocked = 0
        
        rng.shuffle(candidates)
        
        for p in candidates:
            # Check validity
            valid = True
            for idx in curr_indices:
                if S_mask & _C[p][idx]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute new blocked
            new_blocked = blocked
            for idx in curr_indices:
                new_blocked |= _C[p][idx]
            
            score = bin(new_blocked & ~blocked & ~S_mask).count('1')
            
            if score > best_score:
                best_score = score
                best_candidate = p
                best_new_blocked = new_blocked
            elif score == best_score and best_candidate is not None:
                if rng.random() < 0.5:
                    best_candidate = p
                    best_new_blocked = new_blocked
                    
        if best_candidate is None:
            break
            
        S_mask |= (1 << best_candidate)
        blocked = best_new_blocked
        
    return S_mask

def _local_search(S_mask, max_iters=50):
    """Local search to minimize size."""
    for _ in range(max_iters):
        improved = False
        
        # Pass 1: Try removing points
        indices = []
        m = S_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
            
        for idx in list(indices):
            new_mask = S_mask & ~(1 << idx)
            if _is_maximal(new_mask):
                S_mask = new_mask
                improved = True
                # Restart indices
                indices = []
                m = S_mask
                while m:
                    lsb = m & -m
                    idx2 = lsb.bit_length() - 1
                    indices.append(idx2)
                    m ^= lsb
                break # Restart loop after improvement
        
        if improved:
            continue
        
        # Pass 2: Try swaps
        indices = []
        m = S_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
            
        found_swap = False
        for p in indices:
            base_mask = S_mask & ~(1 << p)
            blocked_base = _get_blocked(base_mask)
            
            # Unblocked points
            unblocked = ALL_MASK & ~base_mask & ~blocked_base
            
            m = unblocked
            while m:
                lsb = m & -m
                q = lsb.bit_length() - 1
                m ^= lsb
                
                # Check if q is valid with base_mask
                valid = True
                # We need to check q against all pairs in base_mask
                # Actually, we just need to check if q forms an isosceles triangle with any pair in base_mask
                # This is equivalent to: q is not in C[i][j] for any i,j in base_mask
                # Which is: q is not in blocked_base
                # Wait, blocked_base is the union of C[i][j].
                # So if q is in unblocked, it is NOT in blocked_base.
                # So q is valid with respect to base_mask?
                # No. Validity means no isosceles triangle in the set.
                # The set is base_mask U {q}.
                # Isosceles triangles can be formed by:
                # 1. Two points in base_mask and q.
                #    This is covered by C[i][j] containing q?
                #    C[i][j] contains k if (i,j,k) is isosceles.
                #    So if q is in C[i][j] for some i,j in base_mask, then (i,j,q) is isosceles.
                #    Since q is in unblocked, q is NOT in blocked_base, so q is NOT in any C[i][j] for i,j in base_mask.
                #    So this condition is satisfied.
                # 2. One point in base_mask and two in {q}? No, only one q.
                # 3. All three in base_mask? Already valid.
                # So if q is in unblocked, base_mask U {q} is VALID.
                
                # Now check maximality
                if _is_maximal(base_mask | (1 << q)):
                    S_mask = base_mask | (1 << q)
                    improved = True
                    found_swap = True
                    break
                    
            if found_swap:
                break
                
        if not improved:
            break
            
    return S_mask

def construct():
    _init()
    
    best_size = float('inf')
    best_mask = 0
    start_time = time.time()
    deadline = start_time + 280.0
    
    trial_count = 0
    
    while time.time() < deadline:
        seed = trial_count * 7919 + 12345
        
        S_mask = _greedy_build(seed)
        
        # Local search
        S_mask = _local_search(S_mask, max_iters=50)
        
        size = bin(S_mask).count('1')
        
        if size < best_size:
            best_size = size
            best_mask = S_mask
            if best_size <= 5:
                break
                
        trial_count += 1
    
    # Final polish
    if time.time() < deadline:
        S_mask = _local_search(best_mask, max_iters=100)
        size = bin(S_mask).count('1')
        if size < best_size:
            best_size = size
            best_mask = S_mask
    
    # Convert mask to list of points
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(_points[idx])
        m ^= lsb
    
    return result
