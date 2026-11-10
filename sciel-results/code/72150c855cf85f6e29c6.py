# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _count_convex_7gons(points):
    """
    Count the number of 7-point subsets in convex position using DP.
    """
    n = len(points)
    if n < 7:
        return 0
    
    # Sort points lexicographically
    sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[sorted_indices]
    
    total_count = 0
    
    for i in range(n - 6):
        pivot = sorted_points[i]
        # Points with index > i in sorted order
        remaining = sorted_points[i + 1:]
        m = len(remaining)
        if m < 6:
            continue
            
        # Vectors from pivot to remaining points
        vecs = remaining - pivot
        # Angles
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        # Sort remaining points by angle
        angle_order = np.argsort(angles)
        remaining_sorted = remaining[angle_order]
        
        # Precompute cross products for turns?
        # We need turn(a, b, c) = (b-a)x(c-a) > 0?
        # Contract: (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9
        # This is cross(q-p, r-q).
        # Let's use this definition.
        
        # DP[l][k][j] is expensive memory-wise if we store full 3D array for each i.
        # But m is small (<= 32).
        # We can use a 2D array for current length and update.
        
        # dp[k][j] = number of chains of current length ending at k, j
        # We need to iterate length from 2 to 7.
        
        # Base: length 2 (p_i, q_j)
        # Actually, let's define dp[l][k][j] where k, j are indices in remaining_sorted.
        # l=2: chain p_i, q_j. Only 1 way? No, the "turn" condition is not applied yet?
        # The turn condition is applied when we add a point.
        # Chain p_i, q_a. No turn to check yet?
        # When we add q_b: check turn(p_i, q_a, q_b).
        # So dp[2][a][b] = 1 if turn(p_i, q_a, q_b) > 0 else 0.
        
        # Let's initialize dp for length 2.
        # dp is a 2D array of size m x m.
        # dp[a][b] valid if a < b (angle order) and turn(pivot, remaining_sorted[a], remaining_sorted[b]) > 0.
        
        dp = np.zeros((m, m), dtype=np.int64)
        
        # Precompute turn matrix?
        # turn[a][b] = cross(pivot, remaining_sorted[a], remaining_sorted[b])
        # turn[a][b] = (remaining_sorted[a,0]-pivot[0])*(remaining_sorted[b,1]-remaining_sorted[a,1]) - 
        #             (remaining_sorted[a,1]-pivot[1])*(remaining_sorted[b,0]-remaining_sorted[a,0])
        
        # Actually, the contract formula is:
        # (qx-px)(ry-qy) - (qy-py)(rx-qx)
        # Let p=pivot, q=remaining_sorted[a], r=remaining_sorted[b]
        
        # We can compute this for all a < b.
        # For efficiency, we can compute on the fly in the loop since m is small.
        
        # Initialize dp for length 2
        for a in range(m):
            pa = remaining_sorted[a]
            for b in range(a + 1, m):
                pb = remaining_sorted[b]
                # Check turn(pivot, pa, pb)
                cross = (pa[0] - pivot[0]) * (pb[1] - pa[1]) - (pa[1] - pivot[1]) * (pb[0] - pa[0])
                if cross > 1e-9:
                    dp[a][b] = 1
                    
        # Now extend to length 3, 4, 5, 6, 7
        # Current length is 2. We want to reach length 7 (which means 6 points from remaining).
        # So we need 5 more extensions.
        
        for length in range(3, 8):
            new_dp = np.zeros((m, m), dtype=np.int64)
            for a in range(m):
                for b in range(a + 1, m):
                    # We are extending a chain ending at a, b?
                    # No, the previous state was dp[prev_a][a] where prev_a < a < b.
                    # The chain is ... prev_a, a. We add b.
                    # We need sum over prev_a < a of dp[prev_a][a] * I(turn(remaining_sorted[prev_a], remaining_sorted[a], remaining_sorted[b]) > 0)
                    
                    # Let's optimize this inner loop.
                    # For fixed a, b:
                    # sum_{prev_a < a} dp[prev_a][a] * valid(prev_a, a, b)
                    
                    # We can precompute valid(prev_a, a, b) for fixed a, b?
                    # Or just loop.
                    
                    total = 0
                    pb = remaining_sorted[b]
                    pa = remaining_sorted[a]
                    for prev_a in range(a):
                        if dp[prev_a][a] == 0:
                            continue
                        p_prev = remaining_sorted[prev_a]
                        # Check turn(p_prev, pa, pb)
                        cross = (pa[0] - p_prev[0]) * (pb[1] - pa[1]) - (pa[1] - p_prev[1]) * (pb[0] - pa[0])
                        if cross > 1e-9:
                            total += dp[prev_a][a]
                    new_dp[a][b] = total
                    
            dp = new_dp
            
        # Now dp[a][b] contains counts for chains of length 7 ending at a, b.
        # We need to check the closing turn: turn(remaining_sorted[a], remaining_sorted[b], pivot) > 0.
        # Wait, the chain is p_i, q_{...}, q_a, q_b.
        # The last points are a, b.
        # The closing turn is q_a -> q_b -> p_i.
        # Formula: (q_b - q_a) x (p_i - q_b) ?
        # Contract: (qx-px)(ry-qy) - (qy-py)(rx-qx)
        # Let p=q_a, q=q_b, r=pivot.
        # (q_b[0]-q_a[0])*(pivot[1]-q_b[1]) - (q_b[1]-q_a[1])*(pivot[0]-q_b[0])
        
        for a in range(m):
            for b in range(a + 1, m):
                if dp[a][b] == 0:
                    continue
                pa = remaining_sorted[a]
                pb = remaining_sorted[b]
                cross = (pb[0] - pa[0]) * (pivot[1] - pb[1]) - (pb[1] - pa[1]) * (pivot[0] - pb[0])
                if cross > 1e-9:
                    total_count += dp[a][b]
                    
    return total_count

def _is_valid(points):
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Distance check
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    if np.min(dist2) < 1e-12:
        return False
    
    # Collinearity check
    for i in range(33):
        for j in range(i + 1, 33):
            for k in range(j + 1, 33):
                val = points[i, 0] * (points[j, 1] - points[k, 1]) + \
                      points[j, 0] * (points[k, 1] - points[i, 1]) + \
                      points[k, 0] * (points[i, 1] - points[j, 1])
                if abs(val) < 1e-9:
                    return False
    return True

def erdos_szekeres_problem7() -> np.ndarray:
    n = 33
    np.random.seed(0)
    points = np.random.rand(n, 2)
    
    # Ensure initial validity
    attempts = 0
    while not _is_valid(points) and attempts < 100:
        points = np.random.rand(n, 2)
        attempts += 1
        
    current_count = _count_convex_7gons(points)
    best_points = points.copy()
    best_count = current_count
    
    start_time = time.time()
    time_limit = 580 # Safety margin
    
    while time.time() - start_time < time_limit:
        idx = np.random.randint(n)
        original = points[idx].copy()
        
        # Perturb
        noise = np.random.normal(0, 0.01, 2)
        points[idx] += noise
        
        # Bounds check
        if np.any(points[idx] < 0) or np.any(points[idx] > 1):
            points[idx] = original
            continue
            
        # Quick validity check
        valid = True
        # Distance
        for j in range(n):
            if j == idx: continue
            d2 = np.sum((points[idx] - points[j])**2)
            if d2 < 1e-12:
                valid = False
                break
        if not valid:
            points[idx] = original
            continue
            
        # Collinearity
        for j in range(n):
            if j == idx: continue
            for k in range(n):
                if k == idx or k == j: continue
                if j > k: continue
                val = points[idx, 0] * (points[j, 1] - points[k, 1]) + \
                      points[j, 0] * (points[k, 1] - points[idx, 1]) + \
                      points[k, 0] * (points[idx, 1] - points[j, 1])
                if abs(val) < 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            points[idx] = original
            continue
            
        new_count = _count_convex_7gons(points)
        
        if new_count < best_count:
            best_count = new_count
            best_points = points.copy()
        else:
            points[idx] = original

    return best_points

# EVOLVE-BLOCK-END
