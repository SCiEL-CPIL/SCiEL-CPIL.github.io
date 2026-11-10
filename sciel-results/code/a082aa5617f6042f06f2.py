# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than or equal to their minimum pairwise distance, aiming to maximize the number of points.

    Strategy:
    Use all integer vectors with L1-norm 2 (i.e., exactly two non-zero coordinates, each being +1 or -1).
    
    In dimension d=11:
    - Number of such vectors is C(11, 2) * 4 = 55 * 4 = 220.
    - For any such vector v, ||v||^2 = 2.
    - For any two distinct such vectors u, v:
        - If they share no indices, u.v = 0.
        - If they share one index, u.v = \pm 1.
        - If they share both indices (same support), they are distinct sign combinations.
          The dot product is s1*t1 + s2*t2 where s,t in {1,-1}.
          Possible values: 2 (identical, excluded), 0, -2.
          So max dot product for distinct vectors with same support is 0.
        - Thus, global max dot product is 1.
    - Min squared distance = min (||u||^2 + ||v||^2 - 2 u.v) = 2 + 2 - 2(1) = 2.
    - Max squared norm = 2.
    - Condition: min_dist^2 >= max_norm^2 => 2 >= 2. True.
    
    This configuration is valid and has k=220.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    d = 11
    points_list = []
    
    # Generate all weight-2 vectors
    for i in range(d):
        for j in range(i + 1, d):
            # 1, 1
            v = [0] * d
            v[i] = 1
            v[j] = 1
            points_list.append(v)
            
            # 1, -1
            v = [0] * d
            v[i] = 1
            v[j] = -1
            points_list.append(v)
            
            # -1, 1
            v = [0] * d
            v[i] = -1
            v[j] = 1
            points_list.append(v)
            
            # -1, -1
            v = [0] * d
            v[i] = -1
            v[j] = -1
            points_list.append(v)
            
            if time.time() > deadline - 5:
                # Safety break, though generation is very fast
                break
        if time.time() > deadline - 5:
            break

    points = np.array(points_list, dtype=np.int64)
    
    # Verify validity to be safe (exact integer arithmetic)
    if points.shape[0] < 2:
        return points
        
    norms2 = np.sum(points**2, axis=1)
    max_norm2 = np.max(norms2)
    
    k = points.shape[0]
    min_dist2 = float('inf')
    
    # Check all pairs
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < min_dist2:
                min_dist2 = d2
        if time.time() > deadline - 5:
            # If verification is taking too long (unlikely for k=220), return best effort
            # But we must ensure validity. If we can't finish, we might return invalid.
            # However, k=220 is small.
            pass

    # Check zero vector
    if np.any(np.all(points == 0, axis=1)):
        return points[:0]
        
    if min_dist2 < max_norm2:
        # Fallback to seed (2 points) if logic fails
        return np.array([np.ones(d), -np.ones(d)]).astype(np.int64)
        
    # Round and cast (already ints, but per spec)
    points = np.around(points).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
