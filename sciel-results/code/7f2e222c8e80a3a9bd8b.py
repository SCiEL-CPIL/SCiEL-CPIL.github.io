import numpy as np
import random
import time

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)
POINT_TO_INDEX = {p: i for i, p in enumerate(GRID_POINTS)}

# Precompute squared distances
coords = np.array(GRID_POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diffs * diffs, axis=-1)

def is_no_isosceles(S_indices):
    """
    Check if the set of point indices S_indices forms a no-isosceles set.
    """
    if len(S_indices) < 3:
        return True
    
    # Vectorized check:
    # For each a in S, distances to S\{a} must be unique.
    # We can check this by comparing dist_sq[a, S] for all a in S.
    # Let M be the matrix of distances between points in S.
    # M[i, j] = dist_sq[S[i], S[j]]
    # For each row i, M[i, :] (excluding diagonal) must have unique values.
    
    S_arr = np.array(S_indices)
    M = dist_sq[S_arr][:, S_arr]
    
    # Mask diagonal
    mask = ~np.eye(len(S_arr), dtype=bool)
    
    # For each row, check if non-diagonal elements are unique
    # We can do this by sorting and checking for duplicates, or using a set.
    # Numpy doesn't have a direct "unique per row" that returns a boolean mask easily without loops or fancy indexing.
    # Given small N (<= 16), a loop over rows is fast enough if the inner check is vectorized or simple.
    
    for i in range(len(S_arr)):
        row = M[i, mask[i]]
        if len(np.unique(row)) != len(row):
            return False
            
    return True

def is_maximal(S_indices):
    """
    Check if the set S_indices is maximal.
    """
    S_set = set(S_indices)
    S_arr = np.array(S_indices)
    n_S = len(S_arr)
    
    # Precompute distance sets for each a in S
    # dist_sets[a] = set of dist_sq[a, b] for b in S, b != a
    # We can store these as numpy arrays or sets. Sets are good for membership.
    dist_sets = []
    for i in range(n_S):
        others = np.delete(S_arr, i)
        if len(others) > 0:
            dists = dist_sq[S_arr[i], others]
            dist_sets.append(set(dists))
        else:
            dist_sets.append(set())
            
    # For each q not in S, check if it can be added
    # 1. dists from q to S are unique
    # 2. for each a in S, dist(q, a) not in dist_sets[a]
    
    for q in range(NUM_POINTS):
        if q in S_set:
            continue
            
        # Check 1: unique distances from q to S
        dists_from_q = dist_sq[q, S_arr]
        if len(np.unique(dists_from_q)) != len(dists_from_q):
            continue # q forms isosceles with S (q is apex)
            
        # Check 2: no collision with existing distances
        valid = True
        for i in range(n_S):
            if dists_from_q[i] in dist_sets[i]:
                valid = False
                break
                
        if valid:
            return False # q can be added, so S is NOT maximal
            
    return True

def can_add_to_set(p, S_indices, dist_sets):
    """
    Check if point p can be added to set S_indices.
    dist_sets is a list of sets corresponding to S_indices.
    """
    if not S_indices:
        return True
        
    S_arr = np.array(S_indices)
    
    # Check 1: distances from p to all a in S must be unique
    dists_from_p = dist_sq[p, S_arr]
    if len(np.unique(dists_from_p)) != len(dists_from_p):
        return False
        
    # Check 2: for each a in S, d(p,a) must not be in dist_sets[a]
    for i, a in enumerate(S_indices):
        if dists_from_p[i] in dist_sets[i]:
            return False
            
    return True

def greedy_construct(random_order):
    """
    Greedily construct a maximal no-isosceles set using the given point order.
    Returns (S_indices, dist_sets)
    """
    S = []
    S_set = set()
    dist_sets = []
    
    for p in random_order:
        if p in S_set:
            continue
        
        if can_add_to_set(p, S, dist_sets):
            # Add p to S
            S.append(p)
            S_set.add(p)
            
            # Update dist_sets
            # For each existing a in S, add dist(p, a) to dist_sets[a]
            for i, a in enumerate(S[:-1]):
                d = dist_sq[p, a]
                dist_sets[i].add(d)
            
            # dist_sets[p] = set of distances from p to all other points in S
            if S[:-1]:
                dist_sets.append(set(dist_sq[p, S[:-1]]))
            else:
                dist_sets.append(set())
                
    return S, dist_sets

def _rebuild_dist_sets(S):
    dist_sets = []
    for a in S:
        others = [b for b in S if b != a]
        if others:
            dists = dist_sq[a, others]
            dist_sets.append(set(dists))
        else:
            dist_sets.append(set())
    return dist_sets

def _try_remove(S, dist_sets, deadline):
    """
    Try to remove one point from S while maintaining no-isosceles and maximality.
    """
    S_copy = list(S)
    random.shuffle(S_copy)
    for p in S_copy:
        if time.time() >= deadline:
            break
        S_trial = [x for x in S if x != p]
        if not is_no_isosceles(S_trial):
            continue
        if is_maximal(S_trial):
            new_dist_sets = _rebuild_dist_sets(S_trial)
            return S_trial, new_dist_sets
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Run multiple random restarts
    # The key is to run many fast restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        random_order = list(range(NUM_POINTS))
        random.shuffle(random_order)
        
        S, dist_sets = greedy_construct(random_order)
        
        current_size = len(S)
        
        # Local Search: Try to remove points
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_S, new_dist_sets = _try_remove(S, dist_sets, deadline)
            if new_S is not None:
                S = new_S
                dist_sets = new_dist_sets
                improved = True
                
        current_size = len(S)
        
        if current_size < best_size:
            best_size = current_size
            best_set = S.copy()
            
        # If we find a very small set, we might stop early, 
        # but since we want to maximize probability, we continue until deadline.
        # However, if best_size is 1, 2, or 3, it's likely optimal or near optimal.
        if best_size <= 5:
             # We can try a few more, but it's hard to get smaller than 5 for N=16?
             # Actually, 1 point is maximal? No, 1 point is not maximal because you can add others.
             # Minimal maximal set size is likely > 1.
             pass

    if best_set is None:
        return []
        
    return [GRID_POINTS[i] for i in best_set]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
