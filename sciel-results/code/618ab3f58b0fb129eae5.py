import numpy as np
import time
import random

N = 64
TOTAL_POINTS = N * N

# Precompute all points and distance matrix
all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
all_pts64 = all_points.astype(np.int64)

# Distance matrix: dist_matrix[i, j] is squared distance between point i and point j
# Max dist^2 = 2 * 63^2 = 7938. int32 is safe.
diff = all_pts64[:, np.newaxis, :] - all_pts64[np.newaxis, :, :]
dist_matrix = np.sum(diff ** 2, axis=2).astype(np.int32)
del diff

# Precompute unique distance values for faster set operations
# We can use a sorted array of unique distances for binary search, but numpy isin is okay for small K.
# However, to speed up local search, we might want to precompute some structures.
# For now, we stick with the vectorized isin approach which was effective.

def get_valid_candidates(p_idx_arr, curr_indices, dist_mat):
    """
    Vectorized check for isosceles triangles.
    p_idx_arr: array of candidate indices (M,)
    curr_indices: list of indices in current set (K,)
    dist_mat: full distance matrix (N*N, N*N)
    Returns: boolean mask of same length as p_idx_arr, True if valid.
    """
    M = len(p_idx_arr)
    if M == 0:
        return np.array([], dtype=bool)
    
    K = len(curr_indices)
    if K == 0:
        return np.ones(M, dtype=bool)
    
    curr_arr = np.array(curr_indices, dtype=np.int32)
    
    # D[i, j] = dist(p_idx_arr[i], curr_arr[j])
    # Shape: (M, K)
    D = dist_mat[p_idx_arr[:, np.newaxis], curr_arr[np.newaxis, :]]
    
    # Condition 1: No two distances from p to any two points in S can be equal.
    # This means all distances in row i of D must be unique.
    D_sorted = np.sort(D, axis=1)
    # If any adjacent elements in sorted row are equal, it's isosceles (Apex at p)
    valid_p = np.all(np.diff(D_sorted, axis=1) != 0, axis=1)
    
    if not np.any(valid_p):
        return np.zeros(M, dtype=bool)
    
    valid_mask = valid_p.copy()
    
    # Condition 2: For any point q in S (curr_arr[j]), the distance from p to q 
    # must not equal the distance from q to any other point r in S.
    for j in range(K):
        # Distances from q_j to all other points in S
        others = np.delete(curr_arr, j)
        if len(others) == 0:
            continue
        
        # V_j: distances from q_j to others. Shape (K-1,)
        V_j = dist_mat[curr_arr[j], others]
        
        # D[:, j]: distances from all candidates to q_j. Shape (M,)
        # If any candidate p has d(p, q_j) in V_j, then triangle (p, q_j, r) is isosceles at q_j.
        # We need to ensure d(p, q_j) is NOT in V_j.
        
        # np.isin(D[:, j], V_j) returns True if d(p, q_j) is in V_j.
        is_in = np.isin(D[:, j], V_j)
        valid_mask &= ~is_in
        
    return valid_mask

def local_search(indices, dist_mat, deadline):
    """
    Local search to improve the set size.
    Strategy: 
    1. Greedy add.
    2. Swap operations: Remove 1, try to add 2 (net +1). 
    3. Simulated annealing-like acceptance for neutral moves to escape local minima.
    """
    current = list(indices)
    curr_set = set(current)
    remaining = np.setdiff1d(np.arange(len(dist_mat)), np.array(current))
    
    improved = True
    max_swaps = 200
    swap_count = 0
    
    # Phase 1: Try to add points greedily first
    while len(remaining) > 0 and time.time() < deadline:
        valid_mask = get_valid_candidates(remaining, current, dist_mat)
        if not np.any(valid_mask):
            break
        valid_indices = remaining[valid_mask]
        # Choose the one that conflicts with the fewest future candidates? 
        # For now, random is fast.
        chosen_idx = np.random.choice(valid_indices)
        current.append(chosen_idx)
        remaining = np.delete(remaining, np.where(remaining == chosen_idx)[0])
        
    if len(current) == 0:
        return current

    # Phase 2: Swap operations
    while improved and swap_count < max_swaps and time.time() < deadline:
        improved = False
        swap_count += 1
        
        if len(current) < 2:
            break
            
        # Try a few random removals
        for attempt in range(min(20, len(current))):
            if time.time() > deadline:
                break
                
            rem_idx = np.random.randint(len(current))
            removed = current[rem_idx]
            
            # Create new current without removed
            new_current = current[:rem_idx] + current[rem_idx+1:]
            new_remaining = np.union1d(remaining, np.array([removed]))
            
            # Try to add up to 2 points
            temp_current, temp_remaining = try_add_points(new_current, new_remaining, dist_mat, max_adds=2)
            
            # Accept if we gained net points
            if len(temp_current) > len(current):
                current = temp_current
                remaining = temp_remaining
                improved = True
                break
            # Accept same-size with different composition for diversity (Simulated Annealing lite)
            # This helps escape local minima where we are stuck at size K but could get to K+1 later
            elif len(temp_current) == len(current) and len(temp_current) > 0:
                # 30% chance to accept a neutral move to explore
                if np.random.random() < 0.3:
                    current = temp_current
                    remaining = temp_remaining
                    improved = True
                    break
                    
    return current

def try_add_points(current_indices, remaining, dist_mat, max_adds=5):
    """Try to greedily add more points to the current set."""
    curr = list(current_indices)
    rem = remaining.copy()
    
    for _ in range(max_adds):
        if len(rem) == 0:
            break
        vm = get_valid_candidates(rem, curr, dist_mat)
        if not np.any(vm):
            break
        chosen = np.random.choice(rem[vm])
        curr.append(int(chosen))
        rem = np.delete(rem, np.where(rem == chosen)[0])
        
    return curr, rem

def construct():
    start_time = time.time()
    # Safety margin: 290 seconds (limit 330s)
    deadline = start_time + 290
    
    best_set_indices = []
    best_size = 0
    
    # Initial best can be a simple line or random points
    # Let's start with empty
    
    num_restarts = 2000 # We will break based on time
    
    # Precompute indices for all points
    all_indices = np.arange(TOTAL_POINTS)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        # 1. Greedy Construction
        current_indices = []
        remaining = all_indices.copy()
        
        # Randomize the order of remaining points for diversity
        np.random.shuffle(remaining)
        
        while len(remaining) > 0 and time.time() < deadline - 10:
            # Check which remaining points are compatible with current set
            valid_mask = get_valid_candidates(remaining, current_indices, dist_matrix)
            
            if not np.any(valid_mask):
                break
            
            valid_indices = remaining[valid_mask]
            
            # Choose a random valid index
            chosen_idx = np.random.choice(valid_indices)
            
            current_indices.append(chosen_idx)
            
            # Remove chosen from remaining
            remaining = np.delete(remaining, np.where(remaining == chosen_idx)[0])
            
        # 2. Local Search
        if time.time() < deadline - 5:
            current_indices = local_search(current_indices, dist_matrix, deadline - 5)
            
        if len(current_indices) > best_size:
            best_size = len(current_indices)
            best_set_indices = current_indices[:]
            
    # Convert indices back to coordinates
    if best_set_indices:
        best_coords = all_points[best_set_indices]
        return [(int(x), int(y)) for x, y in best_coords]
    else:
        return []

if __name__ == "__main__":
    # Just for testing, not used by evaluator
    res = construct()
    print(f"Found set of size {len(res)}")
