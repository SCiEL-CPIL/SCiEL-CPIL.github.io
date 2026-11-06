import itertools
import time
import random
import numpy as np

N = 16
V = N * N

def _precompute():
    """Precompute distance matrix and conflict masks."""
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Compute squared distances
    # D[i, j] is squared distance between points[i] and points[j]
    # Using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2)
    
    # C[i][j] is a bitmask of points k such that {i, j, k} is isosceles
    # k != i, k != j
    C = [[0] * V for _ in range(V)]
    
    for i in range(V):
        for j in range(i + 1, V):
            mask = 0
            dij = D[i, j]
            
            # Find k such that D[i, k] == dij
            # We can use numpy to find indices, then convert to mask
            # But doing this for each pair might be slow if not vectorized over pairs.
            # Let's just loop k for each (i,j) pair. 256^3 / 2 ~ 8M iterations.
            # Python loop 8M times might take a few seconds.
            # Let's try to vectorize the mask creation for a fixed (i,j).
            
            # mask1: D[i, k] == dij
            # mask2: D[j, k] == dij
            # mask3: D[i, k] == D[j, k]
            
            # We can compute these masks for all k at once for a given (i,j)
            # But we are in a loop over (i,j).
            
            # Alternative: Precompute for each i, the groups of points by distance.
            # dist_groups[i] = {dist_val: bitmask_of_points}
            
            pass

    # Let's use the dist_groups approach for faster C computation
    dist_groups = [{} for _ in range(V)]
    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            d = D[i, j]
            if d not in dist_groups[i]:
                dist_groups[i][d] = 0
            dist_groups[i][d] |= (1 << j)
    
    for i in range(V):
        for j in range(i + 1, V):
            mask = 0
            dij = D[i, j]
            
            # Points k where D[i, k] == dij
            mask |= dist_groups[i].get(dij, 0)
            # Points k where D[j, k] == dij
            mask |= dist_groups[j].get(dij, 0)
            
            # Points k where D[i, k] == D[j, k]
            # This is trickier to precompute simply.
            # Let's just iterate k for this part?
            # Or compute on the fly.
            
            # Actually, the condition is:
            # d(i,j) == d(i,k) OR d(i,j) == d(j,k) OR d(i,k) == d(j,k)
            
            # The first two are handled by dist_groups.
            # For the third: d(i,k) == d(j,k)
            # This means k is on the perpendicular bisector of ij (in Euclidean sense, but grid is discrete).
            
            # Let's just iterate k for the third condition, or combine.
            # Since V=256, iterating k is 256 steps.
            # Total steps: 256*256/2 * 256 ~ 8M.
            # This is acceptable in Python if optimized?
            # 8M simple ops might take ~1-2 seconds.
            
            # Let's refine:
            # mask1 = dist_groups[i].get(dij, 0)
            # mask2 = dist_groups[j].get(dij, 0)
            
            # For mask3, we need k such that D[i,k] == D[j,k].
            # We can precompute for each pair (i,j) the mask of such k?
            # That's what we are doing.
            
            # Let's just do the loop for mask3.
            for k in range(V):
                if k == i or k == j:
                    continue
                if D[i, k] == D[j, k]:
                    mask |= (1 << k)
            
            # Clear i and j from mask
            mask &= ~(1 << i)
            mask &= ~(1 << j)
            
            C[i][j] = mask
            C[j][i] = mask
            
    return points, D, C

_points, _D, _C = None, None, None

def _init():
    global _points, _D, _C
    if _points is None:
        _points, _D, _C = _precompute()

def _get_blocked_mask(S_mask, C):
    """Calculate the union of conflict masks for all pairs in S."""
    blocked = 0
    # Extract indices from S_mask
    # Since V=256, S_mask is a Python int.
    # We need to iterate over set bits.
    # Efficient way to iterate set bits:
    # while mask:
    #   lsb = mask & -mask
    #   idx = lsb.bit_length() - 1
    #   mask ^= lsb
    
    # But we need pairs.
    # Let's list the indices first.
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
            blocked |= C[indices[i]][indices[j]]
            
    return blocked

def _is_valid(S_mask, C):
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
            # Check if any point in S is in C[i][j]
            if S_mask & C[indices[i]][indices[j]]:
                return False
    return True

def _is_maximal(S_mask, C, V):
    """Check if S is maximal."""
    blocked = _get_blocked_mask(S_mask, C)
    # All points not in S must be blocked
    # All_ones = (1 << V) - 1
    # (blocked | S_mask) == All_ones
    return (blocked | S_mask) == (1 << V) - 1

def _greedy_build(seed, C, V):
    """Greedy construction."""
    rng = random.Random(seed)
    S_mask = 0
    U_mask = (1 << V) - 1
    
    while True:
        blocked = _get_blocked_mask(S_mask, C)
        candidates_mask = U_mask & ~blocked & ~S_mask
        
        if not candidates_mask:
            break
            
        # Find candidates
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
        best_score = -1
        best_candidate = None
        
        rng.shuffle(candidates)
        
        for p in candidates:
            # Check validity
            valid = True
            # Get current indices
            curr_indices = []
            m = S_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                curr_indices.append(idx)
                m ^= lsb
                
            for idx in curr_indices:
                if S_mask & C[p][idx]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute new blocked
            new_blocked = _get_blocked_mask(S_mask | (1 << p), C)
            # Score: number of new points blocked
            # new_blocked & ~blocked
            score = bin(new_blocked & ~blocked).count('1')
            
            if score > best_score:
                best_score = score
                best_candidate = p
            elif score == best_score and best_candidate is not None:
                if rng.random() < 0.5:
                    best_candidate = p
                    
        if best_candidate is None:
            break
            
        S_mask |= (1 << best_candidate)
        U_mask &= ~(1 << best_candidate)
        
    return S_mask

def _local_search(S_mask, C, V, max_iters=100):
    """Local search to minimize size."""
    for _ in range(max_iters):
        improved = False
        indices = []
        m = S_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
            
        # Try removing a point
        for idx in indices:
            new_mask = S_mask & ~(1 << idx)
            if _is_maximal(new_mask, C, V):
                S_mask = new_mask
                improved = True
                break
        
        if improved:
            continue
            
        # Try swapping a point
        # Remove p, add q
        # We need to find q that is not in S, such that S\{p} U {q} is valid and maximal
        for p in indices:
            base_mask = S_mask & ~(1 << p)
            
            # Candidates for q: not in S
            q_candidates = (1 << V) - 1 & ~S_mask
            
            # Iterate q
            m = q_candidates
            found_swap = False
            while m:
                lsb = m & -m
                q = lsb.bit_length() - 1
                m ^= lsb
                
                trial_mask = base_mask | (1 << q)
                
                # Check validity
                if not _is_valid(trial_mask, C):
                    continue
                    
                # Check maximality
                if _is_maximal(trial_mask, C, V):
                    S_mask = trial_mask
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
    C = _C
    V = _points and len(_points) or 256
    
    best_size = float('inf')
    best_mask = 0
    start_time = time.time()
    deadline = start_time + 290.0
    
    trial_count = 0
    
    while time.time() < deadline:
        seed = trial_count * 7919 + 12345
        
        S_mask = _greedy_build(seed, C, V)
        
        # Local search
        S_mask = _local_search(S_mask, C, V, max_iters=20)
        
        size = bin(S_mask).count('1')
        
        if size < best_size:
            best_size = size
            best_mask = S_mask
            if best_size <= 5:
                break
                
        trial_count += 1
        
    # Convert mask to list of points
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(_points[idx])
        m ^= lsb
        
    return result
