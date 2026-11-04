"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

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

def _generate_ring_points(n_outer, n_inner, r_inner, theta_offset):
    """
    Generate points on two concentric rings.
    n_outer + n_inner = 14.
    Outer points are on the unit circle (radius 1).
    Inner points are on a circle of radius r_inner.
    theta_offset is the rotation of the inner ring relative to the outer.
    """
    angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + theta_offset
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _refine_symmetric(pts0, time_limit=5.0):
    """
    Perform a quick Nelder-Mead refinement on the 28 coordinates, 
    starting from a symmetric configuration.
    """
    def _negative_score(params):
        pts = params.reshape(14, 2)
        s = _score(pts)
        if s <= 0:
            return 1e9
        return -s
    
    x0 = pts0.flatten()
    start_time = time.time()
    
    try:
        res = minimize(_negative_score, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 200, 'maxfev': 200})
        
        if res.fun < -0: # Valid improvement
            final_pts = res.x.reshape(14, 2)
            final_score = _score(final_pts)
            if final_score > 0:
                return final_pts, final_score
    except Exception:
        pass
        
    # If refinement fails or doesn't improve, return original
    return pts0, _score(pts0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = -float('inf')
    best_pts = None
    
    # Strategy: Explore different symmetric ring splits (n_outer, n_inner)
    # and optimize (r_inner, theta_offset) for each.
    # Then refine the best candidates in full 28D space.
    
    # Candidate splits: (n_outer, n_inner)
    # We know 7-7 is good, but 8-6, 6-8, 9-5, 5-9, 10-4, 4-10 might be better.
    # We'll focus on splits that are reasonably balanced to keep points in convex position or near it.
    splits = [
        (7, 7),
        (8, 6),
        (6, 8),
        (9, 5),
        (5, 9),
        (10, 4),
        (4, 10),
        (11, 3),
        (3, 11),
        (12, 2),
        (2, 12)
    ]
    
    # For each split, do a coarse grid search over r_inner and theta_offset
    # r_inner range: 0.1 to 0.9
    # theta_offset range: 0 to 2*pi/n_inner (symmetry period)
    
    candidate_seeds = []
    
    for n_out, n_in in splits:
        if time.time() > deadline - 20.0:
            break
            
        # Determine symmetry period for theta_offset
        # If n_out and n_in have common divisors, the symmetry group is larger,
        # but generally the period is 2*pi / gcd(n_out, n_in) or just 2*pi/n_in if we fix outer.
        # Actually, rotating inner ring by 2*pi/n_in maps the set to itself if outer is fixed? No.
        # The set of points is invariant under rotation by 2*pi / gcd(n_out, n_in) if both rings are rotated.
        # But here we fix outer ring angles at 0, 2pi/n_out, etc.
        # So we only vary the relative offset. The offset has period 2*pi/n_in?
        # If we rotate inner ring by 2*pi/n_in, the inner points map to each other.
        # The outer points are fixed. So the configuration is identical.
        # Thus theta_offset in [0, 2*pi/n_in) is sufficient.
        
        theta_max = 2 * np.pi / n_in
        r_vals = np.linspace(0.2, 0.8, 15)
        theta_vals = np.linspace(0, theta_max, 15, endpoint=False)
        
        for r in r_vals:
            if time.time() > deadline - 20.0:
                break
            for theta in theta_vals:
                if time.time() > deadline - 20.0:
                    break
                
                pts = _generate_ring_points(n_out, n_in, r, theta)
                s = _score(pts)
                
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
                
                # Store top candidates for refinement
                # We want to keep a diverse set of good seeds
                if s > best_score * 0.95: # Keep seeds that are within 5% of best
                    candidate_seeds.append(pts.copy())
        
    # Refine the top candidates
    # Sort candidates by score? We didn't store scores. Let's just refine the best few found.
    # Since we updated best_pts in the loop, let's collect a few distinct good ones.
    # A simpler approach: Keep a list of (score, pts) for top 10 unique configurations.
    
    # Re-do: Collect top 10 candidates during the grid search.
    # The above loop didn't do that efficiently. Let's refine the current best and a few others.
    
    # To be safe and efficient, let's just refine the single best point found so far, 
    # and maybe a few random perturbations of it, or points from other splits.
    
    # Let's refine the best point found in the grid search.
    if best_pts is not None:
        if time.time() < deadline - 5.0:
            refined_pts, refined_score = _refine_symmetric(best_pts, time_limit=min(20.0, deadline - time.time() - 2.0))
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts
    
    # Also try refining a few other promising splits if time permits.
    # We'll pick one representative from splits (8,6) and (6,8) if they were evaluated.
    # Since we didn't track them separately, we'll just run a few more random symmetric seeds
    # from the "promising" region of the parameter space.
    
    rng = np.random.default_rng(42)
    for i in range(5):
        if time.time() > deadline - 10.0:
            break
        
        # Pick a random split from the list
        n_out, n_in = rng.choice(splits)
        r = rng.uniform(0.3, 0.7)
        theta = rng.uniform(0, 2*np.pi/n_in)
        
        pts = _generate_ring_points(n_out, n_in, r, theta)
        s = _score(pts)
        
        if s > best_score * 0.9: # Only refine if it's competitive
            if time.time() < deadline - 5.0:
                refined_pts, refined_score = _refine_symmetric(pts, time_limit=min(10.0, deadline - time.time() - 2.0))
                if refined_score > best_score:
                    best_score = refined_score
                    best_pts = refined_pts

    return best_pts
# EVOLVE-BLOCK-END
