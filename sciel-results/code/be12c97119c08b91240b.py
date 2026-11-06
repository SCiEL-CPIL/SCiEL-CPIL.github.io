import time
import numpy as np
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 270.0
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    points_arr = np.array(all_points, dtype=np.int32)
    num_points = len(all_points)
    
    # Precompute squared distance matrix
    # dist_matrix[i, j] = squared distance between points_arr[i] and points_arr[j]
    # Using broadcasting
    # points_arr shape (4096, 2)
    # diff shape (4096, 4096, 2)
    # This might be memory heavy: 4096*4096*2*4 bytes = ~134MB.
    # Let's try to be more memory efficient or just compute on the fly if needed.
    # Given 270s, we can afford to compute it once.
    
    # More efficient distance calculation
    # d^2 = x1^2 + y1^2 + x2^2 + y2^2 - 2(x1x2 + y1y2)
    # Let's just use the standard broadcasting, it's simple.
    try:
        diff = points_arr[:, np.newaxis, :] - points_arr[np.newaxis, :, :]
        dist_matrix = np.sum(diff**2, axis=2)
    except MemoryError:
        # Fallback to on-the-fly calculation if memory is an issue
        dist_matrix = None
        print("Memory error, falling back to slow check")

    best_set = []
    
    def is_valid_addition_vec(p_idx, selected_indices):
        """
        p_idx: index of candidate point
        selected_indices: list of indices of selected points
        """
        k = len(selected_indices)
        if k < 2:
            return True
            
        sel_arr = np.array(selected_indices, dtype=np.int32)
        
        # Distances from p to all selected
        d_sp = dist_matrix[p_idx, sel_arr]
        
        # Check uniqueness of d_sp (Apex at p)
        if len(np.unique(d_sp)) != k:
            return False
            
        # Distances between selected points
        # d_ss[i, j] = dist_matrix[sel_arr[i], sel_arr[j]]
        d_ss = dist_matrix[sel_arr[:, np.newaxis], sel_arr[np.newaxis, :]]
        
        # Check Apex at a_i: d(a_i, a_j) == d(a_i, p)
        # d_ss[i, j] == d_sp[i]
        M1 = (d_ss == d_sp[:, np.newaxis])
        
        # Check Apex at a_j: d(a_i, a_j) == d(a_j, p)
        # d_ss[i, j] == d_sp[j]
        M2 = (d_ss == d_sp[np.newaxis, :])
        
        # Combine
        conflicts = M1 | M2
        
        # Ignore diagonal (i == j)
        np.fill_diagonal(conflicts, False)
        
        return not np.any(conflicts)

    # Multi-start greedy
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        random.seed(attempt * 12345 + 67890)
        
        point_order = list(range(num_points))
        random.shuffle(point_order)
        
        selected_indices = []
        
        for p_idx in point_order:
            if time.time() > deadline:
                break
                
            if is_valid_addition_vec(p_idx, selected_indices):
                selected_indices.append(p_idx)
                
        # Convert indices back to points
        current_set = [all_points[i] for i in selected_indices]
        
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set
# EVOLVE-BLOCK-END
