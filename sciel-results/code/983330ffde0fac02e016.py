import numpy as np
import itertools
import random
import time

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)
POINT_TO_INDEX = {p: i for i, p in enumerate(GRID_POINTS)}

# Precompute squared distances
coords = np.array(GRID_POINTS, dtype=np.int32)
# dist_sq[i, j] = squared distance between point i and point j
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diffs * diffs, axis=-1)

def is_no_isosceles(S_indices):
    """
    Check if the set of point indices S_indices forms a no-isosceles set.
    Condition: For each point a in S, all distances to other points in S must be unique.
    """
    if len(S_indices) < 3:
        return True
    
    S_set = set(S_indices)
    for a in S_indices:
        # Get distances from a to all other points in S
        other_indices = [b for b in S_indices if b != a]
        if not other_indices:
            continue
        dists = dist_sq[a, other_indices]
        # Check for duplicates
        if len(np.unique(dists)) != len(dists):
            return False
    return True

def is_maximal(S_indices):
    """
    Check if the set S_indices is maximal.
    Maximal means no point q not in S can be added without violating no-isosceles.
    """
    S_set = set(S_indices)
    # Precompute distance sets for each a in S to speed up checks?
    # dist_sets[a] = set of dist_sq[a, b] for b in S, b != a
    dist_sets = {}
    for a in S_indices:
        others = [b for b in S_indices if b != a]
        if others:
            dist_sets[a] = set(dist_sq[a, others])
        else:
            dist_sets[a] = set()
    
    for q in range(NUM_POINTS):
        if q in S_set:
            continue
        
        # Check if q can be added
        # 1. Distances from q to all a in S must be unique
        dists_from_q = dist_sq[q, S_indices]
        if len(np.unique(dists_from_q)) != len(dists_from_q):
            continue # q forms isosceles with some pair in S (q is apex)
            
        # 2. For each a in S, d(q,a) must not be in dist_sets[a]
        valid = True
        for a in S_indices:
            if dist_sq[q, a] in dist_sets[a]:
                valid = False
                break
        
        if valid:
            # q can be added, so S is NOT maximal
            return False
            
    return True

def can_add_to_set(p, S_indices, dist_sets):
    """
    Check if point p can be added to set S_indices.
    dist_sets[a] = set of distances from a to other points in S (for a in S)
    Returns True if p can be added.
    """
    if not S_indices:
        return True
    
    # Check 1: distances from p to all a in S must be unique
    dists_from_p = dist_sq[p, S_indices]
    if len(np.unique(dists_from_p)) != len(dists_from_p):
        return False
    
    # Check 2: for each a in S, d(p,a) must not be in dist_sets[a]
    for a in S_indices:
        if dist_sq[p, a] in dist_sets[a]:
            return False
    
    return True

def greedy_construct(random_order):
    """
    Greedily construct a maximal no-isosceles set using the given point order.
    Returns (S_indices, dist_sets) where dist_sets[a] = set of distances from a to S\{a}
    """
    S = []
    S_set = set()
    dist_sets = {}
    
    for p in random_order:
        if p in S_set:
            continue
        
        if can_add_to_set(p, S, dist_sets):
            # Add p to S
            S.append(p)
            S_set.add(p)
            
            # Update dist_sets: add distance from p to each existing point,
            # and add distance from each existing point to p
            for a in S[:-1]:
                d = dist_sq[p, a]
                if a not in dist_sets:
                    dist_sets[a] = set()
                dist_sets[a].add(d)
            
            # dist_sets[p] = set of distances from p to all other points in S
            if S[:-1]:
                dist_sets[p] = set(dist_sq[p, S[:-1]])
            else:
                dist_sets[p] = set()
    
    return S, dist_sets

def try_remove_and_minimize(S, dist_sets, deadline):
    """
    Try to minimize the set by removing points one at a time.
    Returns (new_S, new_dist_sets, improved)
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle S to try removals in different orders
        random.shuffle(S)
        
        for i in range(len(S)):
            p = S[i]
            # Try removing p
            S_trial = S[:i] + S[i+1:]
            
            # Check if S_trial is maximal
            if is_maximal(S_trial):
                # Rebuild dist_sets for S_trial
                new_dist_sets = {}
                for a in S_trial:
                    others = [b for b in S_trial if b != a]
                    if others:
                        new_dist_sets[a] = set(dist_sq[a, others])
                    else:
                        new_dist_sets[a] = set()
                
                S = S_trial
                dist_sets = new_dist_sets
                improved = True
                break # Restart loop with new S
    
    return S, dist_sets, improved

def try_swap(S, dist_sets, deadline):
    """
    Try to improve the set by swapping one point for another.
    Returns (new_S, new_dist_sets, improved)
    """
    improved = False
    S_set = set(S)
    
    # For each point p in S, try removing it and adding a different point q
    for i in range(len(S)):
        if time.time() >= deadline:
            break
        
        p = S[i]
        S_without_p = S[:i] + S[i+1:]
        
        # Try to find a point q not in S that can be added to S_without_p
        # First, build dist_sets for S_without_p
        dist_sets_trial = {}
        for a in S_without_p:
            others = [b for b in S_without_p if b != a]
            if others:
                dist_sets_trial[a] = set(dist_sq[a, others])
            else:
                dist_sets_trial[a] = set()
        
        # Try adding random points q not in S
        candidates = [q for q in range(NUM_POINTS) if q not in S_set]
        random.shuffle(candidates)
        
        for q in candidates[:20]:  # Try at most 20 candidates per removal
            if can_add_to_set(q, S_without_p, dist_sets_trial):
                # Check if S_without_p + [q] is maximal
                S_new = S_without_p + [q]
                if is_maximal(S_new):
                    # Build new dist_sets
                    new_dist_sets = {}
                    for a in S_new:
                        others = [b for b in S_new if b != a]
                        if others:
                            new_dist_sets[a] = set(dist_sq[a, others])
                        else:
                            new_dist_sets[a] = set()
                    
                    return S_new, new_dist_sets, True
    
    return S, dist_sets, improved

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Initial solution: Random Greedy Maximal
    # We will run multiple restarts with different random orders
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # 1. Generate a random maximal set
        random_order = list(range(NUM_POINTS))
        random.shuffle(random_order)
        
        S, dist_sets = greedy_construct(random_order)
        
        current_size = len(S)
        
        # 2. Local Search to minimize size
        # First try removals
        S, dist_sets, _ = try_remove_and_minimize(S, dist_sets, deadline)
        current_size = len(S)
        
        # Then try swaps
        improved = True
        while improved and time.time() < deadline:
            S, dist_sets, improved = try_swap(S, dist_sets, deadline)
            current_size = len(S)
        
        # Update best
        if current_size < best_size:
            best_size = current_size
            best_set = S.copy()
            
        # Early exit if we found a very small set?
        # Theoretical minimum is unknown, but for N=16, it's likely small.
        # If best_size is very small, we can stop.
        if best_size <= 12: # Heuristic threshold
            break

    if best_set is None:
        # Fallback: return an empty set (shouldn't happen, but safe)
        return []
    
    return [GRID_POINTS[i] for i in best_set]

if __name__ == "__main__":
    # Test run
    res = construct()
    print(f"Found set of size {len(res)}")
