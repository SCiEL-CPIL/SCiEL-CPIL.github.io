"""Minimum maximal no-isosceles set on [0,N)^2."""
import random
import time
import itertools
from collections import defaultdict

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute isosceles triples using bitsets for speed
# conflict_mask[i] is a bitmask of points that form an isosceles triangle with point i
# We need: for a chosen set S, a point p can be added iff for all pairs (a,b) in S, p doesn't form iso with a,b
# Equivalently: p is "blocked" by S if there exists a pair (a,b) in S such that (a,b,p) is isosceles

# Precompute: for each pair (i,j), which points k make (i,j,k) isosceles
# Then for a set S, point p is blocked if p is in the union of "bad for pair (a,b)" for all (a,b) in S

# Let's precompute bad_points[i][j] = set of k such that (i,j,k) is isosceles
bad_points = [[set() for _ in range(N_PTS)] for _ in range(N_PTS)]

for i in range(N_PTS):
    for j in range(i+1, N_PTS):
        xi, yi = ALL_POINTS[i]
        xj, yj = ALL_POINTS[j]
        d2 = (xi-xj)**2 + (yi-yj)**2
        for k in range(N_PTS):
            if k == i or k == j:
                continue
            xk, yk = ALL_POINTS[k]
            ik2 = (xi-xk)**2 + (yi-yk)**2
            jk2 = (xj-xk)**2 + (yj-yk)**2
            # Isosceles if any two sides equal
            if d2 == ik2 or d2 == jk2 or ik2 == jk2:
                bad_points[i][j].add(k)
                bad_points[j][i].add(k)

# For faster lookup, convert to bitmasks
bad_mask = [[0] * N_PTS for _ in range(N_PTS)]
for i in range(N_PTS):
    for j in range(N_PTS):
        mask = 0
        for k in bad_points[i][j]:
            mask |= (1 << k)
        bad_mask[i][j] = mask

def _can_add_fast(p_idx, chosen_indices):
    """Check if point p_idx can be added to set of chosen indices."""
    for a in chosen_indices:
        for b in chosen_indices:
            if a >= b:
                continue
            if bad_mask[a][b] & (1 << p_idx):
                return False
    return True

def _is_maximal_fast(chosen_indices):
    """Check if the set is maximal (no point can be added)."""
    chosen_set = set(chosen_indices)
    for p_idx in range(N_PTS):
        if p_idx in chosen_set:
            continue
        if _can_add_fast(p_idx, chosen_indices):
            return False
    return True

def _greedy_random():
    """Random greedy construction."""
    order = list(range(N_PTS))
    random.shuffle(order)
    chosen = []
    for p_idx in order:
        if _can_add_fast(p_idx, chosen):
            chosen.append(p_idx)
    return chosen

def _greedy_priority():
    """Greedy with priority: prefer points that block many others."""
    # Compute for each point, how many other points it "conflicts with"
    # A point p conflicts with q if there exists r such that (p,q,r) is isosceles
    # Actually, let's use a different heuristic: prefer points that are in many isosceles triples
    # This means they block more potential additions
    
    # For each point, count how many pairs (a,b) it forms isosceles with
    point_density = [0] * N_PTS
    for i in range(N_PTS):
        for j in range(i+1, N_PTS):
            for k in bad_points[i][j]:
                point_density[k] += 1
    
    # Sort by density descending, with random tie-breaking
    order = list(range(N_PTS))
    random.shuffle(order)  # Shuffle for randomness
    order.sort(key=lambda x: -point_density[x])
    
    chosen = []
    for p_idx in order:
        if _can_add_fast(p_idx, chosen):
            chosen.append(p_idx)
    return chosen

def _shrink(current, deadline):
    """Try to remove points to get a smaller maximal set."""
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for i in range(len(current)):
            if time.time() > deadline:
                return current
            test = current[:i] + current[i+1:]
            if _is_maximal_fast(test):
                current = test
                improved = True
                break
    return current

def _branch_and_bound(chosen, deadline, best_size):
    """
    Branch and bound to find smaller maximal sets.
    Strategy: try to remove points, using bounds to prune.
    """
    best = list(chosen)
    
    # Lower bound: we need at least some points to be maximal
    # Simple bound: 0 (we can always try to remove)
    
    def _search(current, depth):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current) >= len(best):
            return
        
        # Try removing each point
        for i in range(len(current)):
            if time.time() > deadline:
                return
            test = current[:i] + current[i+1:]
            if _is_maximal_fast(test):
                best = test
                # Continue searching for even smaller
                _search(test, depth + 1)
                return  # Don't continue with other removals at this level
    
    _search(chosen, 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    best = None
    best_size = float('inf')
    
    # Multiple random greedy restarts
    n_restarts = 0
    while time.time() < deadline - 40.0:
        # Alternate between random and priority greedy
        if n_restarts % 2 == 0:
            candidate = _greedy_random()
        else:
            candidate = _greedy_priority()
        
        # Shrink
        improved = _shrink(candidate, deadline - 20.0)
        
        if len(improved) < best_size:
            best_size = len(improved)
            best = improved
            
            # Try branch and bound on the best
            if time.time() < deadline - 30.0:
                bb_result = _branch_and_bound(best, deadline - 15.0, best_size)
                if len(bb_result) < best_size:
                    best_size = len(bb_result)
                    best = bb_result
        
        n_restarts += 1
    
    # Final improvement pass
    if best is not None:
        final = _shrink(best, deadline)
        if len(final) < best_size:
            best = final
            best_size = len(final)
    
    if best is None:
        best = _greedy_random()
    
    return [ALL_POINTS[i] for i in best]
