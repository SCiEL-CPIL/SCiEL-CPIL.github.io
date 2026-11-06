import numpy as np
import random
import time
import math

N = 64
GRID_SIZE = N * N
ALL_POINTS = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)

# Pre-compute squared distances as a NumPy matrix for vectorized operations
# DIST[i, j] = squared distance between point i and point j
DIST = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)

def precompute_distances():
    for i in range(GRID_SIZE):
        x1, y1 = ALL_POINTS[i]
        for j in range(i + 1, GRID_SIZE):
            x2, y2 = ALL_POINTS[j]
            dx = x1 - x2
            dy = y1 - y2
            d_sq = dx * dx + dy * dy
            DIST[i, j] = d_sq
            DIST[j, i] = d_sq

# Run pre-computation at module load time
precompute_distances()

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Helper: Check if a point can be added to a set of indices
    # Returns True if valid, False otherwise
    # Uses vectorized NumPy operations for speed
    def can_add(candidate, current_indices):
        if len(current_indices) == 0:
            return True
        
        # Get distances from candidate to all current points
        d_cand = DIST[candidate, current_indices]
        
        # Check 1: No duplicate distances from candidate
        # If any distance appears more than once, invalid
        if len(np.unique(d_cand)) != len(d_cand):
            return False
        
        # Check 2 & 3: For every pair (a, b) in current, 
        # d(candidate, a) != d(a, b) and d(candidate, b) != d(a, b)
        # This is equivalent to: 
        # For all a, b in current: 
        #   d(c, a) != d(a, b)
        #   d(c, b) != d(a, b)
        
        # Vectorized approach:
        # Create a matrix of distances between all pairs in current_set
        # pair_dists[i, j] = DIST[current[i], current[j]]
        # We need to ensure d_cand[i] != pair_dists[i, j] for all j != i
        # And d_cand[j] != pair_dists[i, j] for all i != j
        
        # Actually, a simpler check:
        # For each a in current:
        #   d(c, a) must not equal d(a, b) for any b in current, b != a
        #   d(c, b) must not equal d(a, b) for any b in current, b != a
        
        # Let's check if d_cand[i] is in the row DIST[current[i], current]
        # excluding the diagonal (distance to self is 0, but d_cand[i] > 0)
        
        # Extract submatrix of distances between current points
        # sub_dist[i, j] = DIST[current[i], current[j]]
        sub_dist = DIST[np.ix_(current_indices, current_indices)]
        
        # For each i, we check if d_cand[i] exists in sub_dist[i, :]
        # But we must be careful: d_cand[i] is DIST[candidate, current[i]]
        # We need to ensure DIST[candidate, current[i]] != DIST[current[i], current[j]] for all j
        
        # Let's do this row by row to avoid huge temporary arrays if possible, 
        # but vectorization is key.
        
        # For each i in current_indices:
        #   d_ci = DIST[candidate, current[i]]
        #   row_i = DIST[current[i], current_indices]
        #   if d_ci in row_i: return False
        
        # Vectorized:
        # Create a matrix where M[i, j] = (d_cand[i] == sub_dist[i, j])
        # If any M[i, j] is True, it's invalid.
        
        # d_cand is shape (m,)
        # sub_dist is shape (m, m)
        # We want to compare d_cand[i] with sub_dist[i, j] for all i, j.
        
        # Expand d_cand to (m, 1) to compare with (m, m)
        d_cand_expanded = d_cand[:, np.newaxis]
        
        # Check if d_cand[i] equals any element in sub_dist[i, :]
        # This finds if d(c, current[i]) == d(current[i], current[j])
        matches = (d_cand_expanded == sub_dist)
        
        # If any match is found, invalid
        if np.any(matches):
            return False
            
        return True

    def can_swap(candidate, current_indices, remove_idx):
        # Create the set without remove_idx
        new_indices = [current_indices[i] for i in range(len(current_indices)) if i != remove_idx]
        return can_add(candidate, new_indices)

    # Simulated Annealing
    current_set = []
    current_set_set = set()
    
    # Initial greedy construction to get a decent starting point
    indices = list(range(GRID_SIZE))
    random.shuffle(indices)
    
    for k in indices:
        if time.time() > deadline - 10:
            break
        if can_add(k, current_set):
            current_set.append(k)
            current_set_set.add(k)
    
    best_size = len(current_set)
    best_set = current_set[:]
    
    # Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-4
    
    # We will use a list for current_set for fast indexing
    # And a set for O(1) membership checks if needed, but we mainly use indices
    
    while time.time() < deadline and T > min_T:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.5 and len(current_set) > 0:
            # Try to remove a point
            remove_idx = random.randint(0, len(current_set) - 1)
            removed_point = current_set[remove_idx]
            
            # Try to add a different point
            # Pick a random point not in current_set
            candidate = random.choice([p for p in range(GRID_SIZE) if p not in current_set_set])
            
            if candidate == removed_point:
                continue
            
            # Check if we can swap
            new_indices = [current_set[i] for i in range(len(current_set)) if i != remove_idx]
            if can_add(candidate, new_indices):
                # Perform swap
                current_set[remove_idx] = candidate
                current_set_set.discard(removed_point)
                current_set_set.add(candidate)
                
                # Swap always keeps size same, so delta_score = 0
                # But it might lead to better future moves. 
                # In pure SA for max size, we usually prefer additions.
                # Let's treat this as a neutral move (accept with prob 1)
                # to explore the space.
                
                # Check if this leads to a state where we can add more?
                # No, SA moves are local.
                
        elif move_type < 0.8 and len(current_set) > 0:
            # Try to remove a point (to allow re-adding or different configuration)
            # This reduces size, so we accept with low probability unless T is high
            remove_idx = random.randint(0, len(current_set) - 1)
            removed_point = current_set[remove_idx]
            
            # Calculate delta: size decreases by 1
            delta_score = -1
            
            # Acceptance probability
            if random.random() < math.exp(delta_score / T):
                current_set.pop(remove_idx)
                current_set_set.discard(removed_point)
                
        else:
            # Try to add a point
            # Pick a random point not in current_set
            # To speed up, we can just pick a random index and check if it's in set
            candidate = random.randint(0, GRID_SIZE - 1)
            if candidate in current_set_set:
                continue
            
            if can_add(candidate, current_set):
                current_set.append(candidate)
                current_set_set.add(candidate)
                
                # Size increases by 1
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
            else:
                # Failed to add, no change
                pass
        
        T *= cooling_rate
        
        # If we have found a very good solution, cool down faster
        if best_size > 150:
            T *= 0.99

    return [ALL_POINTS[i] for i in best_set]
