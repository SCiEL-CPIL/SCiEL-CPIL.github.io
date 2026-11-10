import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 550  # Safety margin
    
    # Standard 220-point solution
    points_list_220 = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * d
                    v[i] = s1
                    v[j] = s2
                    points_list_220.append(v)
    points_220 = np.array(points_list_220, dtype=np.int64)
    
    best_points = points_220
    best_k = 220
    
    # Try to find a larger set using weight-3 vectors
    # Generate all weight-3 vectors
    all_w3 = []
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = [0] * d
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            all_w3.append(v)
                            
    all_w3_arr = np.array(all_w3, dtype=np.int64)
    N = len(all_w3_arr)
    
    # Greedy search with random restarts
    # We can also try to mix in weight 2 vectors if they fit, but likely not worth it
    # given the constraint on weight 2 pairs (dist >= 3 -> dot <= 0).
    
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        # Random permutation
        indices = np.random.permutation(N)
        selected_indices = []
        
        # To speed up, we can maintain the set of selected vectors as a list
        # and check against them.
        # If selected set grows large, checking against all is expensive.
        # But N is small (1320).
        
        for idx in indices:
            v = all_w3_arr[idx]
            if not selected_indices:
                selected_indices.append(idx)
                continue
            
            # Check if v is compatible with all selected
            # v dot u <= 1 for all u in selected
            # selected vectors: all_w3_arr[selected_indices]
            # Dot products: all_w3_arr[selected_indices] @ v
            dots = all_w3_arr[selected_indices] @ v
            if np.max(dots) <= 1:
                selected_indices.append(idx)
                
        if len(selected_indices) > best_k:
            best_k = len(selected_indices)
            best_points = all_w3_arr[selected_indices]
            
        # Optional: Try to add weight 2 vectors to the best set found so far?
        # If best_k is from w3, max norm is 3.
        # w2 vectors need dist >= 3 from w3 and from other w2.
        # This is complex. Skip for now.
        
        if best_k > 300: # If we find something significantly better, maybe stop early?
             # No, keep searching until deadline or if we hit a theoretical bound?
             # 1320 is max.
             pass

    # Return the best found
    return np.around(best_points).astype(np.int64)
