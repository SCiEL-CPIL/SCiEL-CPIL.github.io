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
        
        S = []
        S_set = set()
        
        for p in random_order:
            if p in S_set:
                continue
            # Check if p can be added to current S
            # Reuse logic from is_maximal but for adding to S
            # We need to check if S U {p} is no-isosceles.
            
            # Quick check: distances from p to S must be unique
            if S:
                dists = dist_sq[p, S]
                if len(np.unique(dists)) != len(dists):
                    continue
                
                # Check if d(p,a) conflicts with existing distances in S
                # For each a in S, d(p,a) must not be in {d(a,b) for b in S, b!=a}
                # This is expensive to check dynamically without precomputed sets.
                # Let's just check is_no_isosceles on S + [p]
                # Optimization: Only check if S is small or use incremental check.
                # Given N=16, S is likely < 50. O(|S|^2) is fine.
                
                new_S = S + [p]
                if not is_no_isosceles(new_S):
                    continue
            
            S.append(p)
            S_set.add(p)
            
        # Now S is a maximal no-isosceles set?
        # The greedy construction ensures that we considered all points.
        # If a point was skipped, it was because it formed an isosceles with existing points.
        # So S is maximal by construction.
        
        current_size = len(S)
        
        # 2. Local Search to minimize size
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle S to try removals in different orders
            random.shuffle(S)
            
            for i in range(len(S)):
                p = S[i]
                # Try removing p
                S_trial = S[:i] + S[i+1:]
                
                # Check if S_trial is no-isosceles (it is, subset of no-isosceles)
                # Check if S_trial is maximal
                if is_maximal(S_trial):
                    S = S_trial
                    current_size -= 1
                    improved = True
                    break # Restart loop with new S
            
            # Optional: Try swaps if no removals found
            # This might be too slow. Stick to removals first.
            
        # Update best
        if current_size < best_size:
            best_size = current_size
            best_set = S.copy()
            
        # Early exit if we found a very small set?
        # Theoretical minimum is unknown, but for N=16, it's likely small.
        # If best_size is very small, we can stop.
        if best_size <= 10: # Heuristic threshold
            break

    return [GRID_POINTS[i] for i in best_set]

if __name__ == "__main__":
    # Test run
    res = construct()
    print(f"Found set of size {len(res)}")
