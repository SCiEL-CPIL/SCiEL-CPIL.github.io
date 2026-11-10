import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    
    # Fallback function for the 220 points
    def get_220():
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        points.append(v)
        return np.array(points)

    # Base points: +/- 2 e_i (22 points, norm sq 4)
    base_points = []
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 2
        base_points.append(v)
        v[i] = -2
        base_points.append(v)
        
    # Generate all 4-subsets of {0, ..., 10}
    subsets = list(itertools.combinations(range(d), 4))
    
    # We want a set of subsets with pairwise intersection <= 2
    # Greedy approach
    selected_subsets = []
    
    # To improve greedy, we can shuffle or just rely on the order
    # The order of itertools.combinations is lexicographic.
    # Let's try a few random shuffles if time permits, but standard greedy is fast.
    
    for s in subsets:
        conflict = False
        for s2 in selected_subsets:
            if len(set(s) & set(s2)) >= 3:
                conflict = True
                break
        if not conflict:
            selected_subsets.append(s)
            
        # Time guard
        if time.time() - start_time > 550:
            break

    # Generate points from selected subsets
    # Each subset gives 16 points (signs of 4 coords)
    extra_points = []
    for s in selected_subsets:
        i, j, k, l = s
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            v[i] = signs[0]
            v[j] = signs[1]
            v[k] = signs[2]
            v[l] = signs[3]
            extra_points.append(v)
            
    all_points = np.array(base_points + extra_points)
    
    # If we didn't find enough, or validation fails, return 220
    if len(all_points) <= 220:
        return get_220()

    # Validation
    max_norm_sq = 0
    for p in all_points:
        n = np.dot(p, p)
        if n > max_norm_sq:
            max_norm_sq = n
            
    min_dist_sq = float('inf')
    k = len(all_points)
    for i in range(k):
        for j in range(i + 1, k):
            diff = all_points[i] - all_points[j]
            d_sq = np.dot(diff, diff)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                
    if min_dist_sq < max_norm_sq:
        return get_220()
        
    return all_points
