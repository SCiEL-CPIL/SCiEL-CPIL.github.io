"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 40s safety margin
    
    best_score = -float('inf')
    best_pts = None
    
    # Strategy: 
    # The problem is scale-invariant. We can fix the convex hull to be a regular 
    # 14-gon (which maximizes hull area for a given bounding box, or simply 
    # provides a symmetric baseline). However, the optimal configuration might 
    # not have all points on the hull. 
    # 
    # But, let's look at the "1D radial optimization" hint. 
    # If we place all points on a circle, the convex hull is the polygon formed 
    # by them. The area of a triangle with vertices on a unit circle at angles 
    # a, b, c is 0.5 * |sin(b-a) + sin(c-b) + sin(a-c)| (derived from cross products).
    # Actually, for points on a unit circle, Area = 0.5 * |sin(theta_2 - theta_1) + 
    # sin(theta_3 - theta_2) + sin(theta_1 - theta_3)| is not quite right directly.
    # 
    # Let's use the property that for points on a circle of radius R, 
    # Area(i,j,k) = 0.5 * R^2 * |sin(theta_j - theta_i) + sin(theta_k - theta_j) + sin(theta_i - theta_k)|
    # This is only true if the center is inside the triangle. 
    # 
    # Alternative approach: 
    # The "single ring" strategy implies we should search over angular positions 
    # of 14 points on a circle. 
    # Since the score is min_area / hull_area, and both scale with R^2, the 
    # radius R cancels out. So we can set R=1.
    # 
    # The hull area for 14 points on a unit circle depends on their angular 
    # distribution. If they are evenly spaced, it's a regular 14-gon.
    # If we perturb them, the hull area changes, and the min triangle area changes.
    # 
    # Let's try a grid search on angular deviations from the regular 14-gon.
    # We represent the configuration by 14 angles sorted in [0, 2*pi).
    # We can parameterize by "gaps" between consecutive points. 
    # Let g_i be the gap from point i to i+1 (cyclic). sum(g_i) = 2*pi.
    # 
    # We will perform a coarse-to-fine grid search on the gaps.
    # To make it tractable, we can assume some symmetry or use a random restart 
    # with local refinement in the gap space.
    
    # However, the hint specifically says: "Model the problem as a 1D radial 
    # optimization over a single concentric ring... simplifies the geometry to 
    # a pure angular packing problem".
    
    # Let's try a simple random search in the space of 13 independent gaps 
    # (the 14th is determined).
    
    rng = np.random.default_rng(42)
    
    # Baseline: Regular 14-gon
    angles_base = np.linspace(0, 2*np.pi, 14, endpoint=False)
    pts_base = np.column_stack([np.cos(angles_base), np.sin(angles_base)])
    best_score = _score(pts_base)
    best_pts = pts_base.copy()
    
    # Local search in gap space
    # We define a state by the gaps g[0..12], with g[13] = 2*pi - sum(g[0..12]).
    # We must ensure all gaps > 0.
    
    def _evaluate_gaps(gaps):
        """
        Takes an array of 13 gaps (positive), computes the 14th, 
        reconstructs points, and returns score.
        """
        if np.any(gaps <= 1e-9):
            return -1.0, None
        gap_13 = 2 * np.pi - np.sum(gaps)
        if gap_13 <= 1e-9:
            return -1.0, None
        
        all_gaps = np.append(gaps, gap_13)
        angles = np.cumsum(all_gaps)
        # angles[0] is 0? No, cumsum starts with g[0]. 
        # Let's set angle[0] = 0.
        angles = np.zeros(14)
        angles[1:] = np.cumsum(all_gaps[:13])
        
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        score = _score(pts)
        return score, pts

    # Start from regular gaps
    current_gaps = np.full(13, 2 * np.pi / 14)
    current_score, current_pts = _evaluate_gaps(current_gaps)
    
    if current_score > best_score:
        best_score = current_score
        best_pts = current_pts

    step_size = 0.1
    min_step = 1e-4
    
    # Simple hill climbing in gap space
    # We modify one gap at a time.
    # To keep the sum constant, if we increase gap i, we must decrease another gap j.
    # This maintains the "circle" constraint naturally.
    
    while step_size > min_step:
        improved = False
        for i in range(13):
            if time.time() > deadline:
                break
            
            # Try moving mass from gap j to gap i
            for j in range(13):
                if i == j:
                    continue
                
                # Increase gap i by step, decrease gap j by step
                trial_gaps = current_gaps.copy()
                trial_gaps[i] += step_size
                trial_gaps[j] -= step_size
                
                if trial_gaps[i] <= 0 or trial_gaps[j] <= 0:
                    continue
                    
                s, _ = _evaluate_gaps(trial_gaps)
                
                if s > current_score + 1e-12:
                    current_gaps = trial_gaps
                    current_score = s
                    improved = True
                    # Update best if necessary
                    if current_score > best_score:
                        best_score = current_score
                        # Reconstruct best_pts
                        _, best_pts = _evaluate_gaps(current_gaps)
                    # Restart loops to check other directions with new center
                    break 
            if improved:
                break
                
        if not improved:
            step_size *= 0.5
        else:
            # If improved, we might keep step size or reduce slightly to settle
            # Standard pattern search keeps it. Let's keep it.
            pass
            
        if time.time() > deadline:
            break
            
    # If we have time, try a few random restarts in gap space
    if time.time() < deadline - 5.0:
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            
            # Random initial gaps (Dirichlet distribution ensures positive and sum=1, scale to 2pi)
            raw_gaps = rng.dirichlet(np.ones(14))
            init_gaps_full = raw_gaps * 2 * np.pi
            # We need 13 gaps for our parameterization
            init_gaps = init_gaps_full[:13]
            
            s, pts = _evaluate_gaps(init_gaps)
            if s < 0:
                continue
            
            # Local refine this random start
            cur_gaps = init_gaps.copy()
            cur_score = s
            step = 0.05
            min_step_local = 1e-4
            time_limit_local = min(2.0, deadline - 2.0 - time.time())
            start_local = time.time()
            
            while step > min_step_local:
                improved_local = False
                for i in range(13):
                    if time.time() > start_local + time_limit_local:
                        break
                    for j in range(13):
                        if i == j: continue
                        trial = cur_gaps.copy()
                        trial[i] += step
                        trial[j] -= step
                        if trial[i] <= 0 or trial[j] <= 0: continue
                        s_trial, _ = _evaluate_gaps(trial)
                        if s_trial > cur_score + 1e-12:
                            cur_gaps = trial
                            cur_score = s_trial
                            improved_local = True
                            break
                    if improved_local:
                        break
                if not improved_local:
                    step *= 0.5
                if time.time() > start_local + time_limit_local:
                    break
            
            if cur_score > best_score:
                best_score = cur_score
                _, best_pts = _evaluate_gaps(cur_gaps)
                
    return best_pts
# EVOLVE-BLOCK-END
