import numpy as np
import random
import time
from collections import defaultdict

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)

# Precompute squared distances as numpy array
coords = np.array(GRID_POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diffs * diffs, axis=-1).astype(np.int32)

# Precompute for each point p: sorted unique distances to all other points
# This helps with the "no isosceles" constraint checking
# For point p, d(p, a) for a in S must all be distinct, AND for each a in S,
# d(p,a) must not equal d(a, b) for any b in S, b != a

# dist_sq[p, a] gives squared distance from p to a
# Constraint for no-isosceles:
#   For all distinct a, b, c in S: d(a,b) != d(a,c)  (no isosceles at vertex a)
#   Equivalently: for each a in S, all d(a, s) for s in S\{a} are distinct

def _check_validity(S_indices):
    """Check if a set of point indices forms a valid no-isosceles set.
    Uses numpy for speed."""
    if len(S_indices) <= 1:
        return True
    S = np.array(S_indices)
    # Get all pairwise distances within S
    sub_dist = dist_sq[np.ix_(S, S)]
    # For each row i (point S[i]), all off-diagonal entries must be distinct
    for i in range(len(S)):
        row = sub_dist[i].copy()
        row[i] = -1  # exclude self
        if len(np.unique(row)) != len(row):
            return False
    return True

def _can_add_vectorized(p, S_indices):
    """Check if point p can be added to S_indices.
    Returns True if valid."""
    if len(S_indices) == 0:
        return True
    
    S = np.array(S_indices)
    
    # Check 1: distances from p to all points in S must be distinct
    d_from_p = dist_sq[p, S]
    if len(np.unique(d_from_p)) != len(d_from_p):
        return False
    
    # Check 2: for each a in S, d(p, a) must not equal d(a, b) for any b in S, b != a
    # i.e., d(p,a) must not be in {d(a,b) : b in S, b != a}
    for i, a in enumerate(S):
        d_pa = dist_sq[p, a]
        # Get all distances from a to other points in S
        d_a_to_S = dist_sq[a, S]
        d_a_to_S[i] = -1  # exclude self
        if d_pa in d_a_to_S:
            return False
    
    return True

def _is_maximal(S_indices):
    """Check if S_indices is maximal: no point outside can be added."""
    S_set = set(S_indices)
    for q in range(NUM_POINTS):
        if q in S_set:
            continue
        if _can_add_vectorized(q, S_indices):
            return False
    return True

def _greedy_construct(order):
    """Greedy construction: add points in given order if valid."""
    S = []
    for p in order:
        if _can_add_vectorized(p, S):
            S.append(p)
    return S

def _try_remove(S, p):
    """Try removing point p from S. Returns new S if still maximal, else None."""
    S_new = [x for x in S if x != p]
    if _is_maximal(S_new):
        return S_new
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Phase 1: Multiple greedy constructions with random orders
    # Each greedy is fast due to vectorized checks
    attempts = 0
    while time.time() < deadline - 100:
        attempts += 1
        random_order = list(range(NUM_POINTS))
        random.shuffle(random_order)
        
        S = _greedy_construct(random_order)
        current_size = len(S)
        
        if current_size < best_size:
            best_size = current_size
            best_set = S.copy()
        
        if best_size <= 8:
            break
    
    if best_set is None:
        random_order = list(range(NUM_POINTS))
        random.shuffle(random_order)
        best_set = _greedy_construct(random_order)
        best_size = len(best_set)
    
    # Phase 2: Local search - try removing points
    current = best_set.copy()
    current_size = best_size
    improved = True
    while improved and time.time() < deadline - 50:
        improved = False
        shuffled = current[:]
        random.shuffle(shuffled)
        
        for p in shuffled:
            S_new = _try_remove(current, p)
            if S_new is not None:
                current = S_new
                current_size = len(current)
                improved = True
                break
    
    if current_size < best_size:
        best_size = current_size
        best_set = current.copy()
    
    # Phase 3: Simulated annealing-style search
    # Try random swaps: remove one point, try to add another
    current = best_set.copy()
    current_size = best_size
    T = 50.0  # Temperature
    cooling = 0.99
    
    while time.time() < deadline - 20:
        if current_size < best_size:
            best_size = current_size
            best_set = current.copy()
        
        # Randomly pick a point to remove
        if len(current) == 0:
            break
        
        p = random.choice(current)
        S_without_p = [x for x in current if x != p]
        
        # Try adding random points not in S_without_p
        candidates = [q for q in range(NUM_POINTS) if q not in set(S_without_p)]
        random.shuffle(candidates)
        
        found_swap = False
        for q in candidates[:20]:
            if _can_add_vectorized(q, S_without_p):
                S_new = S_without_p + [q]
                if _is_maximal(S_new):
                    new_size = len(S_new)
                    # Accept if better or with probability based on temperature
                    if new_size <= current_size:
                        current = S_new
                        current_size = new_size
                        found_swap = True
                        break
                    else:
                        # Accept worse with some probability (annealing)
                        delta = new_size - current_size
                        if random.random() < np.exp(-delta / T):
                            current = S_new
                            current_size = new_size
                            found_swap = True
                            break
        
        T *= cooling
    
    # Final check
    if current_size < best_size:
        best_size = current_size
        best_set = current.copy()
    
    # Phase 4: More aggressive local search with multiple restarts
    while time.time() < deadline - 10:
        # Start from best and try more removals
        S = best_set.copy()
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            shuffled = S[:]
            random.shuffle(shuffled)
            for p in shuffled:
                S_new = _try_remove(S, p)
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
