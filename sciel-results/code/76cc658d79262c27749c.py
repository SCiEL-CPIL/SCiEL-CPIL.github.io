import numpy as np
import random
import time

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)

# Precompute squared distances
coords = np.array(GRID_POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diffs * diffs, axis=-1).astype(np.int64)

# For each point p, the set of distances from p to all OTHER points
# (excluding self-distance 0). We use this to check isosceles constraints.
# dist_to_others[p] = sorted list of distances from p to all q != p
dist_to_others = []
for p in range(NUM_POINTS):
    d = dist_sq[p].copy()
    d[p] = -1  # mark self
    dist_to_others.append(d)

# Precompute: for each point p, the number of distinct distances to other points.
# Points with MANY distinct distances are "flexible" (can coexist with more points).
# Points with FEW distinct distances are "rigid" (cause conflicts easily).
num_distinct_dists = np.array([len(np.unique(dist_to_others[p][dist_to_others[p] >= 0])) for p in range(NUM_POINTS)])

# Precompute distance-to-point columns for fast checking
# dist_col[p] = dist_sq[:, p]  (distance from every point to p)
dist_col = dist_sq.T  # dist_col[p][q] = dist_sq[q][p] = distance q to p

# For fast "can add" check:
# To add point p to set S:
#   1. All dist_sq[p, s] for s in S must be distinct
#   2. For each s in S: dist_sq[p, s] must NOT be in {dist_sq[s, t] for t in S, t != s}
#
# We precompute for each point s a boolean array "has_dist[s, d]" = True if s has distance d to some other point
# But distances go up to 2*N^2, so we use a dict per point or a set.

# Actually, let's precompute for each point s, the set of distances to all other points
dist_sets = [set(dist_to_others[s][dist_to_others[s] >= 0].tolist()) for s in range(NUM_POINTS)]

def _can_add_fast(p, S_set, S_list):
    """Check if point p can be added to set S. S_set is a set of indices, S_list is the list."""
    if not S_list:
        return True
    
    # Check 1: distances from p to all in S must be distinct
    dists_from_p = [dist_sq[p, s] for s in S_list]
    if len(set(dists_from_p)) != len(dists_from_p):
        return False
    
    # Check 2: for each s in S, dist_sq[p, s] must not be in dist_sets[s] minus {dist_sq[s, s]=0}
    # Actually dist_sets[s] already excludes self (we set d[p]=-1), but it includes distances to ALL other points,
    # not just those in S. So we need: dist_sq[p, s] not in {dist_sq[s, t] for t in S, t != s}
    for s in S_list:
        d_ps = dist_sq[p, s]
        # Check if d_ps equals dist_sq[s, t] for any t in S, t != s
        for t in S_list:
            if t == s:
                continue
            if dist_sq[s, t] == d_ps:
                return False
    
    return True

def _can_add_fast_v2(p, S_list):
    """Faster check using precomputed structures."""
    if not S_list:
        return True
    
    # Check 1: distances from p to all in S must be distinct
    # Use numpy for this
    S_arr = np.array(S_list)
    dists_from_p = dist_sq[p, S_arr]
    if len(np.unique(dists_from_p)) != len(dists_from_p):
        return False
    
    # Check 2: for each s in S, dist_sq[p, s] must not equal dist_sq[s, t] for any t in S, t != s
    # Vectorize: for each s in S, compute all dist_sq[s, S] and check if dist_sq[p, s] is among them (excluding self)
    # dist_sq[S_arr, S_arr] gives pairwise distances within S
    sub_dist = dist_sq[np.ix_(S_arr, S_arr)]
    # For row i (point S_arr[i]), the off-diagonal entries are distances to other points in S
    # We need: dist_sq[p, S_arr[i]] not in sub_dist[i, [j for j != i]]
    for i in range(len(S_arr)):
        d_ps = dist_sq[p, S_arr[i]]
        # Get all distances from S_arr[i] to other points in S
        row = sub_dist[i].copy()
        row[i] = -1  # exclude self
        if d_ps in row:
            return False
    
    return True

def _is_maximal_fast(S_list):
    """Check if S_list is maximal: no point outside can be added."""
    S_set = set(S_list)
    S_arr = np.array(S_list) if S_list else np.array([], dtype=np.int32)
    
    for q in range(NUM_POINTS):
        if q in S_set:
            continue
        if _can_add_fast_v2(q, S_list):
            return False
    return True

def _greedy_construct(order):
    """Greedy construction: add points in given order if valid."""
    S = []
    S_set = set()
    for p in order:
        if p in S_set:
            continue
        if _can_add_fast_v2(p, S):
            S.append(p)
            S_set.add(p)
    return S

def _try_remove_fast(S_list, p):
    """Try removing point p from S. Returns new S if still maximal, else None."""
    S_new = [x for x in S_list if x != p]
    if _is_maximal_fast(S_new):
        return S_new
    return None

def _distance_spectrum_order():
    """Create point order sorted by distance spectrum.
    Points with many distinct distances are more flexible, so we might want
    to add them first (they're less likely to cause conflicts).
    Alternatively, add rigid points first to constrain the solution early."""
    # Try: add points with FEWER distinct distances first (rigid points constrain early)
    order = list(range(NUM_POINTS))
    order.sort(key=lambda p: num_distinct_dists[p])
    return order

def _random_order():
    """Random point order."""
    order = list(range(NUM_POINTS))
    random.shuffle(order)
    return order

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Phase 1: Multiple greedy constructions with various orderings
    # Use a mix of spectrum-based and random orders
    attempts = 0
    while time.time() < deadline - 120:
        attempts += 1
        # Alternate between spectrum order and random order
        if attempts % 3 == 0:
            order = _distance_spectrum_order()
            # Add some randomness to the spectrum order
            if len(order) > 1:
                swap_idx = random.randint(0, len(order) - 2)
                order[swap_idx], order[swap_idx + 1] = order[swap_idx + 1], order[swap_idx]
        else:
            order = _random_order()
        
        S = _greedy_construct(order)
        current_size = len(S)
        
        if current_size < best_size:
            best_size = current_size
            best_set = S.copy()
        
        if best_size <= 7:
            break
    
    if best_set is None:
        order = _random_order()
        best_set = _greedy_construct(order)
        best_size = len(best_set)
    
    # Phase 2: Local search - try removing points to reduce size while maintaining maximality
    current = best_set.copy()
    current_size = best_size
    improved = True
    while improved and time.time() < deadline - 60:
        improved = False
        shuffled = current[:]
        random.shuffle(shuffled)
        
        for p in shuffled:
            S_new = _try_remove_fast(current, p)
            if S_new is not None:
                current = S_new
                current_size = len(current)
                improved = True
                break
    
    if current_size < best_size:
        best_size = current_size
        best_set = current.copy()
    
    # Phase 3: Multi-start local search
    # Restart from best and try more aggressive removal sequences
    while time.time() < deadline - 20:
        # Start from current best
        S = best_set.copy()
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            shuffled = S[:]
            random.shuffle(shuffled)
            for p in shuffled:
                S_new = _try_remove_fast(S, p)
                if S_new is not None:
                    S = S_new
                    improved = True
                    break
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Phase 4: Try different initial constructions and refine
    while time.time() < deadline - 10:
        # New greedy with random order
        order = _random_order()
        S = _greedy_construct(order)
        
        # Refine with removals
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            shuffled = S[:]
            random.shuffle(shuffled)
            for p in shuffled:
                S_new = _try_remove_fast(S, p)
                if S_new is not None:
                    S = S_new
                    improved = True
                    break
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S.copy()
    
    return [GRID_POINTS[i] for i in best_set]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
