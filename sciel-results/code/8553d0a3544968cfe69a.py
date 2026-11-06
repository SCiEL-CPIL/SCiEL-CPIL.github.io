import itertools
import time
import random
import numpy as np

N = 16
V = N * N

# Precomputation
# dist[i][j] is squared distance
# conflict_mask[i][j] is an integer bitmask of points k such that {i,j,k} is isosceles (k != i, k != j)

def _precompute():
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Compute squared distances
    dist = np.zeros((V, V), dtype=np.int32)
    for i in range(V):
        diff = pts[i] - pts
        dist[i] = np.sum(diff**2, axis=1)
        
    # Precompute conflict masks
    # conflict_mask[i][j] is a Python int with bit k set if {i,j,k} is isosceles
    conflict_mask = [[0] * V for _ in range(V)]
    
    # To speed up, we can group points by distance from i
    # But V=256 is small enough for O(V^3) precomputation? 256^3 = 16M operations.
    # In Python, 16M simple ops might take a few seconds.
    # Let's optimize the inner loop.
    
    for i in range(V):
        for j in range(i + 1, V):
            dij = dist[i][j]
            mask = 0
            # Iterate k
            # We need d(i,k)==dij OR d(j,k)==dij OR d(i,k)==d(j,k)
            # Let's just iterate k
            for k in range(V):
                if k == i or k == j:
                    continue
                dik = dist[i][k]
                djk = dist[j][k]
                if dik == dij or djk == dij or dik == djk:
                    mask |= (1 << k)
            conflict_mask[i][j] = mask
            conflict_mask[j][i] = mask
            
    return points, conflict_mask

_points, _conflict_mask = None, None

def _init():
    global _points, _conflict_mask
    if _points is None:
        _points, _conflict_mask = _precompute()

def _get_blocked_mask(S_indices):
    """
    Returns a bitmask of all points blocked by the set S_indices.
    A point is blocked if it forms an isosceles triangle with any pair in S_indices.
    """
    blocked = 0
    for i in range(len(S_indices)):
        a = S_indices[i]
        for j in range(i + 1, len(S_indices)):
            b = S_indices[j]
            blocked |= _conflict_mask[a][b]
    return blocked

def _is_valid(S_indices):
    """
    Check if S_indices is a valid no-isosceles set.
    For every pair (a,b) in S, no other c in S should be in conflict_mask[a][b].
    """
    S_mask = 0
    for p in S_indices:
        S_mask |= (1 << p)
        
    for i in range(len(S_indices)):
        a = S_indices[i]
        for j in range(i + 1, len(S_indices)):
            b = S_indices[j]
            # If any bit in S_mask is set in conflict_mask[a][b] (excluding a and b), it's invalid
            # Since conflict_mask[a][b] does not contain bits for a and b, we can just check intersection
            if S_mask & _conflict_mask[a][b]:
                return False
    return True

def _is_maximal(S_indices):
    """
    Check if S_indices is maximal.
    Every point not in S_indices must be blocked.
    """
    S_mask = 0
    for p in S_indices:
        S_mask |= (1 << p)
        
    blocked = _get_blocked_mask(S_indices)
    
    # All 256 bits must be set in (S_mask | blocked)
    full_mask = (1 << V) - 1
    return (S_mask | blocked) == full_mask

def _greedy_build(seed):
    """
    Greedy construction of a maximal set.
    """
    rng = random.Random(seed)
    S = []
    S_mask = 0
    blocked = 0
    
    # Initial step: pick a random point to start? Or just start empty.
    # If S is empty, blocked is 0. All points are candidates.
    
    while True:
        # Check if maximal
        if (S_mask | blocked) == ((1 << V) - 1):
            break
            
        # Find candidates: points not in S and not in blocked
        # candidates_mask = ~(S_mask | blocked) & ((1 << V) - 1)
        candidates_mask = ((1 << V) - 1) & ~(S_mask | blocked)
        
        if candidates_mask == 0:
            # No candidates but not maximal? This shouldn't happen if logic is correct.
            # If blocked doesn't cover everything, there must be candidates.
            # Wait, if S is not maximal, there exists p not in S and not blocked.
            # So candidates_mask should not be 0.
            break

        # Evaluate each candidate
        best_score = -1
        best_candidate = -1
        
        # Iterate through set bits in candidates_mask
        cm = candidates_mask
        while cm:
            # Get lowest set bit
            lsb = cm & -cm
            p = lsb.bit_length() - 1
            cm ^= lsb
            
            # Check validity: adding p must not create isosceles with any pair in S
            valid = True
            new_blocked = 0
            for a in S:
                # p must not be in conflict_mask[a][b] for any b in S?
                # Actually, validity of S U {p} means:
                # 1. For any pair (a,b) in S, p is not in conflict_mask[a][b]. (Checked by p not in blocked)
                # 2. For any a in S, the pair (a,p) does not conflict with any other c in S U {p}.
                #    Since S was valid, (a,p) only conflicts with existing points in S if p is in conflict_mask[a][c] for some c in S.
                #    So we need to check if p is in conflict_mask[a][c] for any c in S.
                #    This is equivalent to checking if (conflict_mask[p][a] & S_mask) != 0.
                if _conflict_mask[p][a] & S_mask:
                    valid = False
                    break
                new_blocked |= _conflict_mask[p][a]
            
            if not valid:
                continue
            
            # Score: how many new points are blocked?
            # new_blocked might include points already blocked or in S.
            # We want to maximize new coverage.
            new_coverage = new_blocked & ~(blocked | S_mask)
            score = new_coverage.bit_count()
            
            if score > best_score:
                best_score = score
                best_candidate = p
            elif score == best_score and best_candidate != -1:
                # Random tie-break
                if rng.random() < 0.5:
                    best_candidate = p

        if best_candidate == -1:
            # No valid candidate found?
            # This implies we are stuck in a local maximum of validity but not global maximality?
            # Or just no candidate improves score?
            # If no candidate is valid, we can't add anything.
            # But if not maximal, there must be a point not blocked.
            # If that point is invalid to add, it means adding it creates isosceles.
            # This is a contradiction if "maximal" implies "cannot add any point without breaking validity".
            # Yes, maximal means no point can be added. So if we can't add any valid point, we are maximal.
            # So if best_candidate is -1, we should be maximal.
            # Let's verify maximality.
            if _is_maximal(S):
                break
            else:
                # Error in logic?
                break

        S.append(best_candidate)
        S_mask |= (1 << best_candidate)
        # Update blocked
        for a in S[:-1]:
            blocked |= _conflict_mask[best_candidate][a]

    return S

def _local_search(S, seed, max_iters=20):
    """
    Local search to minimize size while maintaining maximality.
    """
    rng = random.Random(seed)
    S = list(S)
    
    for _ in range(max_iters):
        improved = False
        
        # Try removing a point
        for i in range(len(S)):
            trial = S[:i] + S[i+1:]
            if _is_maximal(trial) and _is_valid(trial):
                S = trial
                improved = True
                break
        
        if improved:
            continue
            
        # Try swapping
        for i in range(len(S)):
            removed = S[i]
            base = S[:i] + S[i+1:]
            
            # Find candidates to add
            # Candidates are points not in base and not blocked by base?
            # No, we just need to find p such that (base U {p}) is valid and maximal.
            
            # Heuristic: try points that were blocked by 'removed'
            # Or just random points?
            # Let's try points that are currently not in base.
            
            # To be efficient, we can check a few random candidates
            candidates = [p for p in range(V) if p not in base]
            rng.shuffle(candidates)
            
            for p in candidates[:20]: # Limit search
                trial = base + [p]
                if _is_valid(trial) and _is_maximal(trial):
                    S = trial
                    improved = True
                    break
            
            if improved:
                break
                
        if not improved:
            break
            
    return S

def construct():
    _init()
    
    best_size = float('inf')
    best_set = None
    start_time = time.time()
    deadline = start_time + 290.0
    
    trial_count = 0
    
    while time.time() < deadline:
        seed = trial_count * 7919 + 12345
        
        S = _greedy_build(seed)
        
        if len(S) < best_size:
            S_improved = _local_search(S, seed, max_iters=10)
            if len(S_improved) < len(S):
                S = S_improved
                
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
            
            if best_size <= 5:
                break
                
        trial_count += 1
        
        # If we are taking too long per trial, reduce iters
        if time.time() - start_time > 250:
             # Do a few more quick trials
             for _ in range(5):
                 if time.time() > deadline: break
                 seed = trial_count * 7919 + 12345
                 S = _greedy_build(seed)
                 if len(S) < best_size:
                     S_improved = _local_search(S, seed, max_iters=2)
                     if len(S_improved) < len(S):
                         S = S_improved
                 if len(S) < best_size:
                     best_size = len(S)
                     best_set = list(S)
                 trial_count += 1
             break

    if best_set is None:
        return []
        
    return [_points[i] for i in best_set]
