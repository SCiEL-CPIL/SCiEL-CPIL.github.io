# EVOLVE-BLOCK-START
import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their 
    maximum squared norm is smaller than or equal to their minimum pairwise squared distance.
    
    This implementation uses the set of all integer points in R^11 with squared norm 2.
    These points have exactly two non-zero coordinates, which are +/-1.
    There are C(11, 2) * 2^2 = 55 * 4 = 220 such points.
    
    For any two distinct points u, v in this set:
    - max squared norm is 2.
    - min squared distance is 2 (occurs when they share one index with the same sign, 
      e.g., (1,1,0..) and (1,-1,0..) -> diff (0,2,0..) dist^2=4? No.
      (1,1,0..) and (1,0,1..) -> diff (0,1,-1..) dist^2=2).
      
    Wait, let's re-verify the min distance for norm-2 vectors.
    u = (1, 1, 0, ..., 0)
    v = (1, -1, 0, ..., 0)
    u - v = (0, 2, 0, ..., 0) -> ||u-v||^2 = 4.
    
    u = (1, 1, 0, ..., 0)
    w = (1, 0, 1, 0, ..., 0)
    u - w = (0, 1, -1, 0, ..., 0) -> ||u-w||^2 = 2.
    
    So min squared distance is 2.
    Max squared norm is 2.
    Condition: min_dist_sq >= max_norm_sq => 2 >= 2. Holds.
    
    Returns:
        points: np.ndarray of shape (220, 11)
    """
    d = 11
    points = []
    
    # Generate all pairs of indices (i, j) with i < j
    for i in range(d):
        for j in range(i + 1, d):
            # Four sign combinations for the two non-zero coordinates
            # (+1, +1), (+1, -1), (-1, +1), (-1, -1)
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = [0] * d
                    p[i] = si
                    p[j] = sj
                    points.append(p)
                    
    points_np = np.array(points, dtype=np.int64)
    
    # Verify condition to be safe
    if points_np.shape[0] < 2:
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        
    norms_sq = np.sum(points_np**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    min_dist_sq = float('inf')
    n_points = len(points_np)
    
    # Optimization: We know the answer is 220 and it's valid.
    # But the task requires the function to return a valid array.
    # The verification loop is O(k^2 * d) = 220^2 * 11 ~ 500,000 ops, which is fast.
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            diff = points_np[i] - points_np[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                # Early exit if we find a violation? No, we need min.
                # But if min_dist_sq < max_norm_sq, it's invalid.
                # We can break early if we just want to check validity.
                # However, we are confident.
                
    if min_dist_sq >= max_norm_sq:
        return points_np
    else:
        # Fallback to a smaller valid set if something went wrong (should not happen)
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)

# EVOLVE-BLOCK-END
