import itertools
import time
import random
import numpy as np
from collections import defaultdict

N = 16
V = N * N
ALL_MASK = (1 << V) - 1

_points = None
_conflict_mask = None
_conflict_mask_flat = None  # 1D array for faster access
_dist_matrix = None

def _precompute():
    global _points, _conflict_mask, _conflict_mask_flat, _dist_matrix
    
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    _points = points
    
    # Compute squared distances
    dist = np.zeros((V, V), dtype=np.int32)
    for i in range(V):
        diff = pts[i] - pts
        dist[i] = np.sum(diff**2, axis=1)
    _dist_matrix = dist
    
    # Precompute conflict masks
    # conflict_mask[i] is a Python int with bit k set if {i,j,k} is isosceles for some j in S
    # Actually, we need: for a pair (i,j), which points k form isosceles with i,j?
    # But for the set S, we need: for each pair (i,j) in S, no k in S can form isosceles.
    # So for a candidate set S, we check all pairs.
    
    # Let's precompute: for each point p, and for each other point q,
    # the mask of points r such that {p,q,r} is isosceles.
    # This is conflict_mask[p][q] in the original, but let's optimize.
    
    # Actually, let's use a different approach for the GA:
    # Represent the solution as a set of points.
    # The constraint is: for all triples (i,j,k) in S, they are NOT isosceles.
    # Equivalently: for all pairs (i,j) in S, no k in S is in conflict_mask[i][j].
    
    # Let's precompute conflict_mask as a list of lists for clarity,
    # but also create a flat structure for speed.
    
    conflict_mask = [[0] * V for _ in range(V)]
    
    for i in range(V):
        for j in range(i + 1, V):
            dij = dist[i][j]
            mask = 0
            for k in range(V):
                if k == i or k == j:
                    continue
                dik = dist[i][k]
                djk = dist[j][k]
                if dik == dij or djk == dij or dik == djk:
                    mask |= (1 << k)
            conflict_mask[i][j] = mask
            conflict_mask[j][i] = mask
            
    _conflict_mask = conflict_mask
    
    # Create a flat list for faster access: index = i * V + j
    _conflict_mask_flat = [0] * (V * V)
    for i in range(V):
        for j in range(V):
            _conflict_mask_flat[i * V + j] = conflict_mask[i][j]

def _init():
    global _points, _conflict_mask, _conflict_mask_flat, _dist_matrix
    if _points is None:
        _precompute()

def _is_valid_set(S):
    """Check if S is a valid no-isosceles set."""
    S_mask = 0
    for p in S:
        S_mask |= (1 << p)
    
    for i in range(len(S)):
        a = S[i]
        for j in range(i + 1, len(S)):
            b = S[j]
            if S_mask & _conflict_mask[a][b]:
                return False
    return True

def _is_maximal(S):
    """Check if S is maximal (no point can be added)."""
    S_mask = 0
    for p in S:
        S_mask |= (1 << p)
    
    # Compute blocked mask
    blocked = 0
    for i in range(len(S)):
        a = S[i]
        for j in range(i + 1, len(S)):
            b = S[j]
            blocked |= _conflict_mask[a][b]
    
    # If any point is not in S and not blocked, S is not maximal
    if (ALL_MASK & ~(S_mask | blocked)) != 0:
        return False
    return True

def _get_unblocked_points(S):
    """Return list of points that can be added to S."""
    S_mask = 0
    for p in S:
        S_mask |= (1 << p)
    
    blocked = 0
    for i in range(len(S)):
        a = S[i]
        for j in range(i + 1, len(S)):
            b = S[j]
            blocked |= _conflict_mask[a][b]
    
    candidates_mask = ALL_MASK & ~(S_mask | blocked)
    unblocked = []
    cm = candidates_mask
    while cm:
        lsb = cm & -cm
        p = lsb.bit_length() - 1
        unblocked.append(p)
        cm ^= lsb
    return unblocked

def _greedy_build(seed):
    """Greedy construction of a maximal set."""
    rng = random.Random(seed)
    S = []
    S_mask = 0
    blocked = 0
    
    while True:
        if (S_mask | blocked) == ALL_MASK:
            break
            
        candidates_mask = ALL_MASK & ~(S_mask | blocked)
        if candidates_mask == 0:
            break

        best_score = -1
        best_candidate = -1
        
        cm = candidates_mask
        while cm:
            lsb = cm & -cm
            p = lsb.bit_length() - 1
            cm ^= lsb
            
            valid = True
            new_blocked = 0
            for a in S:
                if _conflict_mask[p][a] & S_mask:
                    valid = False
                    break
                new_blocked |= _conflict_mask[p][a]
            
            if not valid:
                continue
            
            new_coverage = new_blocked & ~(blocked | S_mask)
            score = new_coverage.bit_count()
            
            if score > best_score:
                best_score = score
                best_candidate = p
            elif score == best_score and best_candidate != -1:
                if rng.random() < 0.5:
                    best_candidate = p

        if best_candidate == -1:
            break

        S.append(best_candidate)
        S_mask |= (1 << best_candidate)
        for a in S[:-1]:
            blocked |= _conflict_mask[best_candidate][a]

    return S

def _make_maximal(S, seed):
    """Ensure S is maximal by adding points greedily."""
    rng = random.Random(seed)
    S = list(S)
    S_mask = 0
    for p in S:
        S_mask |= (1 << p)
    
    blocked = 0
    for i in range(len(S)):
        a = S[i]
        for j in range(i + 1, len(S)):
            b = S[j]
            blocked |= _conflict_mask[a][b]
    
    while (S_mask | blocked) != ALL_MASK:
        candidates_mask = ALL_MASK & ~(S_mask | blocked)
        if candidates_mask == 0:
            break
        
        # Pick a random candidate that is valid
        cm = candidates_mask
        count = cm.bit_count()
        r = rng.randrange(count)
        tmp = cm
        for _ in range(r):
            lsb = tmp & -tmp
            tmp ^= lsb
        p = (tmp & -tmp).bit_length() - 1
        
        # Check validity
        valid = True
        new_blocked = 0
        for a in S:
            if _conflict_mask[p][a] & S_mask:
                valid = False
                break
            new_blocked |= _conflict_mask[p][a]
        
        if not valid:
            continue
        
        S.append(p)
        S_mask |= (1 << p)
        for a in S[:-1]:
            blocked |= _conflict_mask[p][a]
    
    return S

def _local_search(S, seed, time_limit=10.0):
    """Local search to minimize |S| while maintaining maximality."""
    rng = random.Random(seed)
    S = list(S)
    S_mask = 0
    for p in S:
        S_mask |= (1 << p)
    
    start_time = time.time()
    best_S = list(S)
    best_size = len(S)
    
    while (time.time() - start_time) < time_limit:
        # Try to remove a point
        if len(S) > 1:
            idx = rng.randrange(len(S))
            p_remove = S[idx]
            S_temp = S[:idx] + S[idx+1:]
            S_mask_temp = S_mask & ~(1 << p_remove)
            
            # Check if S_temp is still valid
            valid = True
            for i in range(len(S_temp)):
                a = S_temp[i]
                for j in range(i + 1, len(S_temp)):
                    b = S_temp[j]
                    if S_mask_temp & _conflict_mask[a][b]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Make it maximal
                S_new = _make_maximal(S_temp, seed + idx)
                if len(S_new) < best_size:
                    best_size = len(S_new)
                    best_S = list(S_new)
                    S = list(S_new)
                    S_mask = 0
                    for p in S:
                        S_mask |= (1 << p)
                    continue
        
        # Try to swap: remove one, add another
        if len(S) > 1:
            idx = rng.randrange(len(S))
            p_remove = S[idx]
            S_temp = S[:idx] + S[idx+1:]
            S_mask_temp = S_mask & ~(1 << p_remove)
            
            # Find candidates to add
            blocked_temp = 0
            for i in range(len(S_temp)):
                a = S_temp[i]
                for j in range(i + 1, len(S_temp)):
                    b = S_temp[j]
                    blocked_temp |= _conflict_mask[a][b]
            
            candidates_mask = ALL_MASK & ~(S_mask_temp | blocked_temp)
            if candidates_mask != 0:
                cm = candidates_mask
                count = cm.bit_count()
                r = rng.randrange(count)
                tmp = cm
                for _ in range(r):
                    lsb = tmp & -tmp
                    tmp ^= lsb
                p_add = (tmp & -tmp).bit_length() - 1
                
                # Check if adding p_add is valid
                valid = True
                for a in S_temp:
                    if _conflict_mask[p_add][a] & S_mask_temp:
                        valid = False
                        break
                
                if valid:
                    S_new = S_temp + [p_add]
                    S_new = _make_maximal(S_new, seed + idx + 1000)
                    if len(S_new) < best_size:
                        best_size = len(S_new)
                        best_S = list(S_new)
                        S = list(S_new)
                        S_mask = 0
                        for p in S:
                            S_mask |= (1 << p)
        
        # Try to add a point (even if it increases size, might enable better swaps later)
        # This is less useful, so we skip it to save time.
        
        # Occasionally restart with a slightly perturbed version
        if rng.random() < 0.05:
            # Remove a random point and make maximal
            if len(S) > 1:
                idx = rng.randrange(len(S))
                p_remove = S[idx]
                S_temp = S[:idx] + S[idx+1:]
                S_mask_temp = S_mask & ~(1 << p_remove)
                
                valid = True
                for i in range(len(S_temp)):
                    a = S_temp[i]
                    for j in range(i + 1, len(S_temp)):
                        b = S_temp[j]
                        if S_mask_temp & _conflict_mask[a][b]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    S_new = _make_maximal(S_temp, seed + 5000)
                    if len(S_new) < best_size:
                        best_size = len(S_new)
                        best_S = list(S_new)
                        S = list(S_new)
                        S_mask = 0
                        for p in S:
                            S_mask |= (1 << p)
    
    return best_S

def construct():
    _init()
    
    best_size = float('inf')
    best_set = None
    start_time = time.time()
    deadline = start_time + 290.0
    
    trial_count = 0
    
    while time.time() < deadline:
        seed = trial_count * 7919 + 12345
        
        # Get initial greedy set
        S_greedy = _greedy_build(seed)
        
        # Run local search
        sa_time = 8.0 if time.time() - start_time < 200 else 2.0
        S_ls = _local_search(S_greedy, seed, time_limit=sa_time)
        
        # Ensure maximality
        S_final = _make_maximal(S_ls, seed + 20000)
        
        if len(S_final) < best_size:
            best_size = len(S_final)
            best_set = list(S_final)
            
            if best_size <= 5:
                break
        
        trial_count += 1
    
    if best_set is None:
        return []
        
    return [_points[i] for i in best_set]
