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
    
    # Compute squared distances using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2)
    
    # C[i][j] is a bitmask of points k such that {i, j, k} is isosceles
    C = [[0] * V for _ in range(V)]
    
    # Precompute dist_groups for fast lookup
    dist_groups = [{} for _ in range(V)]
    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            d = int(D[i, j])
            if d not in dist_groups[i]:
                dist_groups[i][d] = 0
            dist_groups[i][d] |= (1 << j)
    
    # Precompute perpendicular bisector masks
    # perp_mask[i][j] = bitmask of k where D[i,k] == D[j,k]
    perp_mask = [[0] * V for _ in range(V)]
    for i in range(V):
        for j in range(i + 1, V):
            mask = 0
            # Vectorized: find k where D[i,k] == D[j,k]
            eq = D[i] == D[j]
            for k in range(V):
                if k != i and k != j and eq[k]:
                    mask |= (1 << k)
            perp_mask[i][j] = mask
            perp_mask[j][i] = mask
    
    # Build conflict masks
    for i in range(V):
        for j in range(i + 1, V):
            mask = 0
            dij = int(D[i, j])
            
            # Points k where D[i, k] == dij
            mask |= dist_groups[i].get(dij, 0)
            # Points k where D[j, k] == dij  
            mask |= dist_groups[j].get(dij, 0)
            # Points k where D[i, k] == D[j, k]
            mask |= perp_mask[i][j]
            
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

def _get_indices(mask):
    """Extract indices from a bitmask."""
    indices = []
    m = mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        indices.append(idx)
        m ^= lsb
    return indices

def _get_blocked_mask(S_mask, C):
    """Calculate the union of conflict masks for all pairs in S."""
    blocked = 0
    indices = _get_indices(S_mask)
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            blocked |= C[indices[i]][indices[j]]
    return blocked

def _is_valid(S_mask, C):
    """Check if S is a valid no-isosceles set."""
    indices = _get_indices(S_mask)
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            if S_mask & C[indices[i]][indices[j]]:
                return False
    return True

def _is_maximal(S_mask, C, V):
    """Check if S is maximal."""
    blocked = _get_blocked_mask(S_mask, C)
    return (blocked | S_mask) == (1 << V) - 1

def _greedy_build(seed, C, V, strategy='blocking'):
    """Greedy construction with multiple strategies."""
    rng = random.Random(seed)
    S_mask = 0
    U_mask = (1 << V) - 1
    
    while True:
        blocked = _get_blocked_mask(S_mask, C)
        candidates_mask = U_mask & ~blocked & ~S_mask
        
        if not candidates_mask:
            break
            
        candidates = _get_indices(candidates_mask)
        
        if not candidates:
            break
            
        best_score = -1
        best_candidate = None
        best_newly_blocked = 0
        
        # Shuffle for randomization
        rng.shuffle(candidates)
        
        # Get current indices for scoring
        curr_indices = _get_indices(S_mask)
        
        for p in candidates:
            # Check validity against current set
            valid = True
            for idx in curr_indices:
                if S_mask & C[p][idx]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute new blocked
            new_blocked = _get_blocked_mask(S_mask | (1 << p), C)
            newly_blocked = new_blocked & ~blocked
            
            if strategy == 'blocking':
                score = bin(newly_blocked).count('1')
            elif strategy == 'random':
                score = rng.random()
            elif strategy == 'balanced':
                # Balance between blocking and leaving options
                remaining = bin(candidates_mask & ~newly_blocked).count('1')
                score = bin(newly_blocked).count('1') * 1.0 - remaining * 0.1
            else:
                score = bin(newly_blocked).count('1')
            
            if score > best_score or (score == best_score and rng.random() < 0.3):
                best_score = score
                best_candidate = p
                best_newly_blocked = newly_blocked
        
        if best_candidate is None:
            break
            
        S_mask |= (1 << best_candidate)
        U_mask &= ~(1 << best_candidate)
        
    return S_mask

def _local_search(S_mask, C, V, max_iters=50, temperature=1.0):
    """Local search to minimize size with multiple move types."""
    rng = random.Random(42)
    best_mask = S_mask
    best_size = bin(S_mask).count('1')
    
    for it in range(max_iters):
        if time.time() > _deadline:
            break
            
        indices = _get_indices(S_mask)
        n = len(indices)
        improved = False
        
        # Move 1: Try removing a point (if still maximal)
        for idx in indices:
            new_mask = S_mask & ~(1 << idx)
            if _is_maximal(new_mask, C, V):
                S_mask = new_mask
                improved = True
                break
        
        if improved:
            continue
        
        # Move 2: Try swapping one point
        for p in indices:
            base_mask = S_mask & ~(1 << p)
            q_candidates = (1 << V) - 1 & ~S_mask
            
            found_swap = False
            m = q_candidates
            while m:
                lsb = m & -m
                q = lsb.bit_length() - 1
                m ^= lsb
                
                trial_mask = base_mask | (1 << q)
                
                if not _is_valid(trial_mask, C):
                    continue
                    
                if _is_maximal(trial_mask, C, V):
                    S_mask = trial_mask
                    improved = True
                    found_swap = True
                    break
                    
            if found_swap:
                break
        
        if improved:
            continue
        
        # Move 3: Try removing two, adding one (2-for-1 move)
        found_21 = False
        for i in range(n):
            for j in range(i + 1, n):
                base_mask = S_mask & ~(1 << indices[i]) & ~(1 << indices[j])
                q_candidates = (1 << V) - 1 & ~S_mask
                
                m = q_candidates
                while m:
                    lsb = m & -m
                    q = lsb.bit_length() - 1
                    m ^= lsb
                    
                    trial_mask = base_mask | (1 << q)
                    
                    if not _is_valid(trial_mask, C):
                        continue
                        
                    if _is_maximal(trial_mask, C, V):
                        S_mask = trial_mask
                        improved = True
                        found_21 = True
                        break
                        
                if found_21:
                    break
            if found_21:
                break
        
        if improved:
            continue
        
        # Move 4: Simulated annealing - accept worse solutions with probability
        if temperature > 0.01:
            # Try a random swap
            if n > 0:
                p = indices[rng.randint(0, n - 1)]
                base_mask = S_mask & ~(1 << p)
                q_candidates = (1 << V) - 1 & ~S_mask
                q_list = _get_indices(q_candidates)
                
                if q_list:
                    q = q_list[rng.randint(0, len(q_list) - 1)]
                    trial_mask = base_mask | (1 << q)
                    
                    if _is_valid(trial_mask, C) and _is_maximal(trial_mask, C, V):
                        # Same size, accept with some probability for diversity
                        if rng.random() < 0.3:
                            S_mask = trial_mask
                        temperature *= 0.95
                        continue
        
        if not improved:
            break
        
        # Update best
        if bin(S_mask).count('1') < best_size:
            best_size = bin(S_mask).count('1')
            best_mask = S_mask
        
        temperature *= 0.95
    
    return best_mask

def _intensive_search(S_mask, C, V, time_limit=30):
    """Intensive local search with more iterations and diverse moves."""
    rng = random.Random(123)
    best_mask = S_mask
    best_size = bin(S_mask).count('1')
    start = time.time()
    
    while time.time() - start < time_limit:
        indices = _get_indices(S_mask)
        n = len(indices)
        improved = False
        
        # Try all single removals
        for idx in indices:
            new_mask = S_mask & ~(1 << idx)
            if _is_maximal(new_mask, C, V):
                S_mask = new_mask
                improved = True
                break
        
        if improved:
            if bin(S_mask).count('1') < best_size:
                best_size = bin(S_mask).count('1')
                best_mask = S_mask
            continue
        
        # Try all swaps
        found_swap = False
        for p in indices:
            base_mask = S_mask & ~(1 << p)
            q_candidates = (1 << V) - 1 & ~S_mask
            q_list = _get_indices(q_candidates)
            
            for q in q_list:
                trial_mask = base_mask | (1 << q)
                
                if not _is_valid(trial_mask, C):
                    continue
                    
                if _is_maximal(trial_mask, C, V):
                    S_mask = trial_mask
                    improved = True
                    found_swap = True
                    break
                    
            if found_swap:
                break
        
        if improved:
            if bin(S_mask).count('1') < best_size:
                best_size = bin(S_mask).count('1')
                best_mask = S_mask
            continue
        
        # Try 2-for-1 moves
        found_21 = False
        for i in range(n):
            for j in range(i + 1, n):
                base_mask = S_mask & ~(1 << indices[i]) & ~(1 << indices[j])
                q_candidates = (1 << V) - 1 & ~S_mask
                q_list = _get_indices(q_candidates)
                
                for q in q_list:
                    trial_mask = base_mask | (1 << q)
                    
                    if not _is_valid(trial_mask, C):
                        continue
                        
                    if _is_maximal(trial_mask, C, V):
                        S_mask = trial_mask
                        improved = True
                        found_21 = True
                        break
                        
                if found_21:
                    break
            if found_21:
                break
        
        if improved:
            if bin(S_mask).count('1') < best_size:
                best_size = bin(S_mask).count('1')
                best_mask = S_mask
            continue
        
        # No improvement, break
        break
    
    return best_mask

_deadline = 0.0

def construct():
    global _deadline
    _init()
    C = _C
    V = len(_points)
    
    best_size = float('inf')
    best_mask = 0
    start_time = time.time()
    _deadline = start_time + 290.0
    
    trial_count = 0
    strategies = ['blocking', 'random', 'balanced']
    
    while time.time() < _deadline:
        seed = trial_count * 7919 + 12345
        strategy = strategies[trial_count % len(strategies)]
        
        S_mask = _greedy_build(seed, C, V, strategy=strategy)
        
        # Quick local search
        S_mask = _local_search(S_mask, C, V, max_iters=30)
        
        size = bin(S_mask).count('1')
        
        if size < best_size:
            best_size = size
            best_mask = S_mask
            
            # If we found a good solution, do intensive search
            if best_size <= 8 and time.time() < _deadline - 40:
                S_mask_int = _intensive_search(S_mask, C, V, time_limit=25)
                size_int = bin(S_mask_int).count('1')
                if size_int < best_size:
                    best_size = size_int
                    best_mask = S_mask_int
            
            if best_size <= 5:
                break
        
        trial_count += 1
        
        # Adaptive: if stuck, try more diverse approaches
        if trial_count > 20 and best_size > 10:
            # Try more random strategies
            for _ in range(3):
                if time.time() >= _deadline - 30:
                    break
                S_alt = _greedy_build(trial_count * 13 + 1000 + _, C, V, strategy='random')
                S_alt = _local_search(S_alt, C, V, max_iters=20)
                size_alt = bin(S_alt).count('1')
                if size_alt < best_size:
                    best_size = size_alt
                    best_mask = S_alt
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
