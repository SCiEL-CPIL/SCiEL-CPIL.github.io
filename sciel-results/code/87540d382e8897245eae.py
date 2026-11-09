import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 14

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    n = points.shape[0]
    
    # Calculate minimum triangle area
    min_area = np.inf
    
    # Loop over all triples
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                # Cross product magnitude gives 2 * area
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    
    if not np.isfinite(min_area) or min_area <= 0:
        return 0.0

    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 0:
        return 0.0

    return min_area / hull_area

def _generate_ring_points(k_inner, k_outer, r_in, r_out, alpha, beta):
    """
    Generate points for a two-ring configuration.
    k_inner points on circle radius r_in, angle offset alpha.
    k_outer points on circle radius r_out, angle offset beta.
    If k_inner == 1, the point is at the center (0,0) effectively, but we can treat it as radius 0.
    Actually, if k_inner == 1, it's better to just place it at origin to maximize convexity?
    Or is it better to have it on a small ring?
    Usually, for N=14, the optimal is likely 1 center + 13 outer, or 2+12, etc.
    Let's assume the "inner" ring points are evenly spaced.
    """
    points = []
    
    if k_inner > 0:
        angles_in = 2 * np.pi * np.arange(k_inner) / k_inner + alpha
        if k_inner == 1:
            # Single inner point: place at center? Or on a small ring?
            # If it's on a ring of radius r_in, it's just one point.
            # For k=1, the "angle" doesn't matter for the point's position relative to itself,
            # but it matters for the symmetry of the whole set.
            # However, a single point on a circle is just a point.
            # Let's assume if k_inner==1, we fix it at (0,0) to maximize hull area?
            # No, the hull is determined by the outer points.
            # Placing the inner point at the centroid (0,0) is usually optimal for maximizing min distance to others?
            # Let's try r_in as a parameter even for k=1, but if k=1, the point is at (r_in*cos(alpha), r_in*sin(alpha)).
            # If we want it at center, r_in=0.
            pass
        
        x_in = r_in * np.cos(angles_in)
        y_in = r_in * np.sin(angles_in)
        points.append(np.column_stack((x_in, y_in)))
    
    if k_outer > 0:
        angles_out = 2 * np.pi * np.arange(k_outer) / k_outer + beta
        x_out = r_out * np.cos(angles_out)
        y_out = r_out * np.sin(angles_out)
        points.append(np.column_stack((x_out, y_out)))
        
    if not points:
        return np.zeros((0, 2))
        
    return np.vstack(points)

def _ring_score(params, k_inner, k_outer):
    """
    Objective function for ring optimization.
    params: [r_in, r_out, alpha, beta]
    """
    r_in, r_out, alpha, beta = params
    
    # Ensure r_in < r_out and r_in >= 0
    # If k_inner == 1, r_in can be 0.
    # If k_inner > 1, r_in should be > 0 to avoid degeneracy?
    # Actually, if r_in == 0 and k_inner > 1, all inner points collapse to one point -> invalid (min area 0).
    # So we must enforce r_in > epsilon if k_inner > 1.
    # But we can just let the optimizer handle it, or add a penalty.
    # Let's add a small penalty if r_in is too small for k_inner > 1.
    
    if k_inner > 1 and r_in < 1e-6:
        r_in = 1e-6
        
    if r_out < r_in:
        # Swap if needed or just return bad score
        # Actually, we define r_out as the larger one.
        # Let's just swap in the generation step or enforce bounds.
        # For simplicity, we assume r_out > r_in in the parameter space.
        pass

    pts = _generate_ring_points(k_inner, k_outer, r_in, r_out, alpha, beta)
    
    if pts.shape[0] != N:
        return 1.0 # Bad score (maximize score, so minimize -score)
        
    score = _calculate_score(pts)
    return -score

def construct():
    start_time = time.time()
    timeout = 135.0 # Safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Define splits of N=14 into inner and outer rings
    splits = []
    for k_in in range(0, N + 1):
        k_out = N - k_in
        # We consider k_in <= k_out to avoid duplicates? 
        # Actually, the roles are different (radius). 
        # But (k_in, k_out) with r_in < r_out is distinct from (k_out, k_in) with r_in < r_out?
        # No, (2, 12) with r1<r2 is the same geometry as (12, 2) with r2>r1?
        # No, the set of points is the same. The labels "inner" and "outer" are just for parameterization.
        # So we only need to test k_in from 0 to N//2?
        # Wait, if k_in=1, k_out=13.
        # If k_in=13, k_out=1. This is the same set of points if we swap radii?
        # Yes. So we can just test k_in from 0 to 7.
        
        # However, k_in=0 means all on one ring.
        # k_in=1 means 1 point, 13 on ring.
        
        splits.append((k_in, k_out))
        
    # To be safe and cover all, let's just test k_in from 1 to 7.
    # k_in=0 is just a regular polygon, which we can initialize separately.
    
    # 1. Initialize with Regular Polygon (k_in=0)
    angles = 2 * np.pi * np.arange(N) / N
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _calculate_score(init_points)
    best_points = init_points.copy()
    
    # 2. Optimize Ring Configurations
    # Parameters: r_in, r_out, alpha, beta
    # Bounds:
    # r_in: [0, 1] (if k_in > 0)
    # r_out: [1, 2] (must be > r_in)
    # alpha: [0, 2*pi]
    # beta: [0, 2*pi]
    
    # We will use differential evolution for the 4 parameters for each split.
    # This is fast because 4D is small.
    
    for k_in, k_out in splits:
        if k_in + k_out != N:
            continue
            
        # Skip if k_in == 0 (handled by init)
        if k_in == 0:
            continue
            
        # Define bounds
        # If k_in == 1, r_in can be 0.
        # If k_in > 1, r_in should be > 0.
        # Let's use [0, 1] for r_in and [1, 3] for r_out to ensure separation.
        # Actually, we can just use [0, 5] for both and let the optimizer sort it out?
        # No, better to enforce r_out > r_in.
        
        # We'll use a wrapper that ensures r_out > r_in.
        def ring_opt_func(params, k_in_val, k_out_val):
            r_in, r_out, alpha, beta = params
            # Enforce r_out > r_in
            if r_out <= r_in:
                r_out = r_in + 1e-6
            if r_in < 0:
                r_in = 0
            if k_in_val > 1 and r_in < 1e-4:
                r_in = 1e-4
            return _ring_score([r_in, r_out, alpha, beta], k_in_val, k_out_val)

        # Initial guess
        # r_in ~ 0.5, r_out ~ 1.5
        x0 = [0.5, 1.5, 0.0, 0.0]
        
        bounds = [(0, 1.5), (1.5, 3.0), (0, 2*np.pi), (0, 2*np.pi)]
        
        # Use Powell for local refinement from a few random starts?
        # Or DE for global?
        # DE on 4D is very fast.
        
        try:
            # Check time
            if time.time() - start_time > timeout - 10:
                break
                
            res_de = differential_evolution(
                ring_opt_func, 
                bounds, 
                args=(k_in, k_out),
                seed=42, 
                maxiter=50, 
                tol=1e-6, 
                polish=False,
                popsize=10
            )
            
            if res_de.success or res_de.fun < -best_score:
                # Evaluate the best found
                r_in, r_out, alpha, beta = res_de.x
                if r_out <= r_in:
                    r_out = r_in + 1e-6
                if k_in > 1 and r_in < 1e-4:
                    r_in = 1e-4
                    
                pts = _generate_ring_points(k_in, k_out, r_in, r_out, alpha, beta)
                s = _calculate_score(pts)
                
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
                    
                    # Refine with Powell
                    if time.time() - start_time < timeout - 20:
                        def local_func(p):
                            r_in_l, r_out_l, alpha_l, beta_l = p
                            if r_out_l <= r_in_l:
                                r_out_l = r_in_l + 1e-6
                            if k_in > 1 and r_in_l < 1e-4:
                                r_in_l = 1e-4
                            return _ring_score([r_in_l, r_out_l, alpha_l, beta_l], k_in, k_out)
                            
                        res_powell = minimize(
                            local_func, 
                            res_de.x, 
                            method='Powell', 
                            bounds=bounds,
                            options={'maxiter': 100, 'disp': False, 'xtol': 1e-8, 'ftol': 1e-8}
                        )
                        
                        if res_powell.fun < -best_score:
                            r_in_f, r_out_f, alpha_f, beta_f = res_powell.x
                            if r_out_f <= r_in_f:
                                r_out_f = r_in_f + 1e-6
                            if k_in > 1 and r_in_f < 1e-4:
                                r_in_f = 1e-4
                                
                            pts_f = _generate_ring_points(k_in, k_out, r_in_f, r_out_f, alpha_f, beta_f)
                            s_f = _calculate_score(pts_f)
                            
                            if s_f > best_score:
                                best_score = s_f
                                best_points = pts_f.copy()
                                
        except Exception:
            pass

    # 3. Final Polish on the best points found using full 28D optimization?
    # This might break the ring symmetry but could find a better non-symmetric solution.
    # However, the ring solution is likely very close to optimal.
    # Let's try a short Powell on the full 28D coordinates if time permits.
    
    if best_points is not None and time.time() - start_time < timeout - 10:
        def full_neg_score(flat):
            pts = flat.reshape(N, 2)
            return -_calculate_score(pts)
            
        try:
            res_full = minimize(
                full_neg_score, 
                best_points.flatten(), 
                method='Powell', 
                options={'maxiter': 50, 'disp': False, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            if res_full.fun < -best_score:
                pts_full = res_full.x.reshape(N, 2)
                s_full = _calculate_score(pts_full)
                if s_full > best_score:
                    best_score = s_full
                    best_points = pts_full
        except Exception:
            pass

    return best_points
