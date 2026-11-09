import random
import time
import numpy as np

N = 64
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_POINTS = N * N

# Precompute squared distances
# dist_matrix[i, j] = squared distance between POINTS[i] and POINTS[j]
dist_matrix = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
for i in range(NUM_POINTS):
    xi, yi = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        xj, yj = POINTS[j]
        d2 = (xi - xj) ** 2 + (yi - yj) ** 2
        dist_matrix[i, j] = d2
        dist_matrix[j, i] = d2

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = []
    best_size = 0
    
    indices = list(range(NUM_POINTS))
    
    # Helper to check if adding p_idx to current_set is valid
    # current_set: list of indices
    # dist_sets: list of sets of squared distances from current_set[i] to others
    def try_add(p_idx, current_set, dist_sets):
        if not current_set:
            return True, []
        
        d_to_p = []
        for i, A_idx in enumerate(current_set):
            d2 = dist_matrix[p_idx, A_idx]
            d_to_p.append(d2)
            # Check if this distance is already used by A to another point
            if d2 in dist_sets[i]:
                return False, None
        
        # Check if distances from P to all existing points are distinct
        if len(set(d_to_p)) != len(d_to_p):
            return False, None
            
        return True, d_to_p

    def perform_add(p_idx, current_set, dist_sets, d_to_p):
        # Update dist_sets for existing points
        for i, A_idx in enumerate(current_set):
            dist_sets[i].add(d_to_p[i])
        
        # Add new point
        current_set.append(p_idx)
        dist_sets.append(set(d_to_p))

    def perform_remove(p_idx, current_set, dist_sets):
        # Find index of p_idx in current_set
        pos = current_set.index(p_idx)
        
        # Get distances from p_idx to all others
        d_to_others = dist_sets[pos]
        
        # Remove these distances from the dist_sets of the other points
        for i, A_idx in enumerate(current_set):
            if i == pos:
                continue
            # The distance is symmetric
            d2 = dist_matrix[p_idx, A_idx]
            dist_sets[i].discard(d2)
            
        # Remove the point and its dist_set
        current_set.pop(pos)
        dist_sets.pop(pos)

    # Initial Greedy Construction to get a good starting point
    def greedy_init(seed):
        random.seed(seed)
        candidates = indices[:]
        random.shuffle(candidates)
        
        current_set = []
        dist_sets = []
        
        for p_idx in candidates:
            valid, d_to_p = try_add(p_idx, current_set, dist_sets)
            if valid:
                perform_add(p_idx, current_set, dist_sets, d_to_p)
                
        return current_set, dist_sets

    # Run initial greedy to get a baseline
    initial_set, initial_dist_sets = greedy_init(42)
    best_size = len(initial_set)
    best_set = initial_set[:]
    
    current_set = initial_set
    dist_sets = initial_dist_sets
    
    # Local Search / Simulated Annealing
    # We will try to add random points. If it fails, we might remove a conflicting point.
    # To diversify, we also do random restarts of the local search phase if stuck.
    
    max_iterations = 500000 # High limit, but time is the real limit
    
    for iter_num in range(max_iterations):
        if time.time() > deadline:
            break
            
        # 1. Try to add a random point not in the set
        # To make this efficient, we can just pick a random index and check if it's in the set
        # Since set membership is O(1) with a set of indices, let's maintain a set of indices in current_set
        # But current_set is a list. Let's use a set for fast lookup.
        # We'll maintain current_set_indices as a set
        
        # Actually, let's just pick a random candidate from all points.
        # If it's already in the set, skip.
        # Probability of picking an existing point is K/4096. If K=100, it's 2.5%. Not bad.
        
        p_idx = random.randint(0, NUM_POINTS - 1)
        
        if p_idx in current_set:
            # If we pick an existing point, maybe try to remove it?
            # Or just skip. Skipping is fine, we'll pick a new one next iter.
            # But to allow "moves", let's sometimes remove a random point from the set
            if random.random() < 0.1 and current_set:
                # Remove a random point from current_set
                remove_idx = random.choice(current_set)
                perform_remove(remove_idx, current_set, dist_sets)
            continue

        valid, d_to_p = try_add(p_idx, current_set, dist_sets)
        
        if valid:
            perform_add(p_idx, current_set, dist_sets, d_to_p)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
        else:
            # If invalid, we can try to resolve the conflict.
            # A simple strategy: remove a random point from the current set that is "conflicting".
            # We don't easily know which one is conflicting without re-checking.
            # Alternative: Just do a random kick.
            # Occasionally, restart the local search from a perturbed state.
            
            # Let's implement a "kick" every N iterations or if stuck for a while.
            # For simplicity, let's just do random removals to allow exploration.
            if random.random() < 0.05 and current_set:
                remove_idx = random.choice(current_set)
                perform_remove(remove_idx, current_set, dist_sets)
                
            # If we are stuck (size hasn't increased in a long time), do a bigger kick
            # We can track last_improvement_iter
            # For now, the random removals should help escape local minima.
            
            # Additional Strategy: Try to swap.
            # Pick a random point in set, remove it, then try to add the failed point again?
            # Or try to add a different point.
            # The current loop structure already tries new points.
            
            # Let's add a more aggressive kick: if we fail 100 times in a row, remove 5 points.
            # We need a counter.
            # To keep code simple, let's just rely on the 5% removal chance.
            # If best_size is small, we might need more aggressive restarts.
            
            # Let's check if we should do a full restart of the local search phase
            # If current_size is much smaller than best_size, restart from best?
            # No, best_set is static.
            # If current_size < best_size - 10, restart current_set from best_set?
            # That might be too expensive to copy.
            
            # Let's just continue.

    return [POINTS[i] for i in best_set]
# EVOLVE-BLOCK-END
