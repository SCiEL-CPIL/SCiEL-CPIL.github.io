import random
import time
import numpy as np

N = 64

def construct():
    # Precompute all points
    all_points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
    
    def greedy_fill(points_array):
        selected_indices = []
        selected_arr = np.empty((0, 2), dtype=np.int32)
        global_dists = set()
        
        # Shuffle indices to randomize order
        indices = np.arange(len(points_array))
        np.random.shuffle(indices)
        
        for idx in indices:
            p = points_array[idx]
            
            if len(selected_indices) == 0:
                selected_indices.append(idx)
                selected_arr = p.reshape(1, -1).copy()
                continue
            
            # Vectorized distance calculation
            # selected_arr: (K, 2), p: (2,)
            diffs = selected_arr - p
            dists = np.sum(diffs * diffs, axis=1)
            
            # Check for duplicates in dists (P as apex)
            # np.unique returns sorted unique values
            if len(np.unique(dists)) < len(dists):
                continue
            
            # Check intersection with global_dists (P as base vertex)
            # Convert dists to list for iteration, or use set
            # set(dists) is O(K)
            # Intersection is O(min(K, len(global_dists)))
            dists_set = set(dists.tolist())
            if dists_set & global_dists:
                continue
            
            # Valid
            selected_indices.append(idx)
            selected_arr = np.vstack((selected_arr, p.reshape(1, -1)))
            global_dists.update(dists_set)
            
        return selected_indices

    start_time = time.time()
    time_limit = 290.0
    
    best_points = []
    
    while time.time() - start_time < time_limit:
        # Shuffle points
        shuffled_points = all_points.copy()
        np.random.shuffle(shuffled_points)
        
        indices = greedy_fill(shuffled_points)
        
        if len(indices) > len(best_points):
            best_points = [tuple(shuffled_points[i]) for i in indices]
            
    return best_points
