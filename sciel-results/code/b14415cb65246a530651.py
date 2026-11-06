"""No-isosceles set construction with optimized validation and local search."""
import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 285  # 285 second deadline to be safe
    
    best_points = []
    best_count = 0
    
    # Fast vectorized validity check
    def is_valid_addition(points, p):
        if not points:
            return True
        
        px, py = p
        P = np.array(points, dtype=np.int64)
        n = len(P)
        
        # Squared distances from p to all points in P
        d_p = (P[:, 0] - px)**2 + (P[:, 1] - py)**2
        
        # Check 1: d(p, a) == d(p, b) for any a != b
        # This is equivalent to checking if there are duplicate values in d_p
        # Using unique is often faster than set() for numpy arrays
        if len(np.unique(d_p)) != n:
            return False
            
        # Squared distances between all pairs in P
        # D_sq[i, j] = ||P_i - P_j||^2
        P_norm_sq = np.sum(P**2, axis=1) # (N,)
        Dot = P @ P.T # (N, N)
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        # D_sq is symmetric, diagonal is 0
        
        # Check 2 and 3: d(p, a) == d(a, b) or d(p, b) == d(a, b)
        # For each pair (i, j) with i < j:
        #   d_p[i] != D_sq[i, j] and d_p[j] != D_sq[i, j]
        
        # Vectorized check using broadcasting
        # cond1[i, j] is True if D_sq[i, j] == d_p[i]
        cond1 = (D_sq == d_p[:, np.newaxis])
        # cond2[i, j] is True if D_sq[i, j] == d_p[j]
        cond2 = (D_sq == d_p[np.newaxis, :])
        
        # We only care about pairs (i, j) with i != j.
        # The diagonal (i == j) has D_sq[i, i] = 0.
        # d_p[i] is distance from p to P_i. Since p is not in P, d_p[i] > 0.
        # So D_sq[i, i] (0) will never equal d_p[i] (>0).
        # So we can just check the whole matrix, but we must exclude the diagonal to be strictly correct about "pairs of distinct points".
        # Actually, if i=j, the condition is d(p, P_i) == d(P_i, P_i) = 0. Since p != P_i, d(p, P_i) > 0, so this is False.
        # So checking the whole matrix is safe, but let's use a mask to be precise and potentially faster if we can short-circuit? No, full matrix is fine.
        
        invalid_mask = cond1 | cond2
        
        # We need to ensure that for any distinct i, j, the condition is not met.
        # Set diagonal to False to ignore self-pairs, though they are already False.
        np.fill_diagonal(invalid_mask, False)
        
        if np.any(invalid_mask):
            return False
        
        return True

    # Greedy construction with random restarts
    def greedy_construct():
        pts = []
        candidates = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(pts, p):
                pts.append(p)
        return pts

    # Local search to improve
    def local_search(initial_pts):
        pts = list(initial_pts)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try adding a random point
            for _ in range(3000):
                if time.time() > deadline:
                    break
                p = (random.randint(0, N-1), random.randint(0, N-1))
                if p not in pts and is_valid_addition(pts, p):
                    pts.append(p)
                    improved = True
                    
            # Phase 2: Try swapping
            if len(pts) > 0:
                for _ in range(300):
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, len(pts) - 1)
                    old_pt = pts[idx]
                    pts_temp = pts[:idx] + pts[idx+1:]
                    
                    # Try to find a replacement
                    for _ in range(100):
                        if time.time() > deadline:
                            break
                        p = (random.randint(0, N-1), random.randint(0, N-1))
                        if p not in pts_temp and is_valid_addition(pts_temp, p):
                            pts = pts_temp + [p]
                            improved = True
                            break
        return pts

    # Run multiple greedy constructions
    for trial in range(80):
        if time.time() > deadline - 10:
            break
        
        pts = greedy_construct()
        if len(pts) > best_count:
            best_points = pts
            best_count = len(pts)
            
        # Apply local search if we have time and potential
        if len(pts) > best_count - 3 and time.time() < deadline - 25:
            improved_pts = local_search(pts)
            if len(improved_pts) > best_count:
                best_points = improved_pts
                best_count = len(improved_pts)
                
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
