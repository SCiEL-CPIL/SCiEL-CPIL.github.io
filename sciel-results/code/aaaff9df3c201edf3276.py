"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(points):
    """Compute the score (radius) for a given set of points."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_boundary)
    
    # Pairwise distances
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _cyclic_coordinate_descent(points, r, max_iter=2000, tol=1e-12):
    """
    Cyclic coordinate descent with hard constraint projections.
    We iterate through each point and adjust it to satisfy constraints with all other points
    and the boundary. This is a direct feasibility search for radius r.
    """
    points = points.copy()
    n = len(points)
    
    # Precompute pairs for efficiency? No, just iterate.
    # We want to find a configuration where min_dist >= 2r and |p_i| <= 1-r.
    
    for iteration in range(max_iter):
        moved = False
        for i in range(n):
            p_i = points[i].copy()
            
            # Check boundary constraint
            norm_i = np.linalg.norm(p_i)
            if norm_i > 1.0 - r:
                # Project onto boundary circle
                if norm_i > 0:
                    p_i = p_i * (1.0 - r) / norm_i
                else:
                    p_i = np.array([1.0 - r, 0.0])
                moved = True
            
            # Check pairwise constraints with all j != i
            # We can do a few passes of projection within the i-loop to resolve conflicts
            # But standard CCD updates one coordinate at a time. Here we update the whole vector p_i.
            # To make it more effective, we can iterate a few times inside for i to resolve
            # conflicts with multiple neighbors.
            
            for _ in range(10): # Inner iterations for this point
                p_i_new = p_i
                for j in range(n):
                    if i == j:
                        continue
                    p_j = points[j]
                    d = np.linalg.norm(p_i_new - p_j)
                    if d < 2.0 * r:
                        # Project p_i_new to be at distance 2r from p_j
                        if d < 1e-12:
                            # Random small displacement
                            angle = np.random.rand() * 2 * np.pi
                            p_i_new = p_j + 2.0 * r * np.array([np.cos(angle), np.sin(angle)])
                        else:
                            direction = (p_i_new - p_j) / d
                            p_i_new = p_j + 2.0 * r * direction
                        # Re-check boundary after projection? 
                        # If we project away from j, we might go out of bounds.
                        # We will handle boundary in the next check or a final clamp.
                
                # Final boundary clamp for this inner iteration
                norm_new = np.linalg.norm(p_i_new)
                if norm_new > 1.0 - r:
                    if norm_new > 0:
                        p_i_new = p_i_new * (1.0 - r) / norm_new
                    else:
                        p_i_new = np.array([1.0 - r, 0.0])
                
                if np.linalg.norm(p_i_new - p_i) < tol:
                    break
                p_i = p_i_new
            
            if np.linalg.norm(p_i - points[i]) > tol:
                points[i] = p_i
                moved = True
        
        if not moved:
            break
            
    return points

def _is_feasible(points, r):
    """Check if points form a valid packing with radius r."""
    if np.any(np.linalg.norm(points, axis=1) > 1.0 - r + 1e-9):
        return False
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 2.0 * r - 1e-9:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # 1. Generate initial candidates
    # Hexagonal lattice
    def get_hex_init(r):
        # Estimate number of points in hex lattice with spacing 2r
        # Area approx pi. Number of points ~ pi / (sqrt(3)*r^2)
        # We need N points.
        # Let's just generate a large grid and pick N closest to center? 
        # Or generate specifically.
        spacing = 2.0 * r
        pts = []
        k = 10
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                x = i * spacing + (j % 2) * spacing / 2.0
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= (1.0 - r)**2:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) < N:
            return None
        # Pick N points. To maximize symmetry, pick those closest to center?
        # Or random subset? 
        # Sorting by radius ensures we fill the center first, which is good for packing.
        radii = np.linalg.norm(pts, axis=1)
        indices = np.argsort(radii)
        return pts[indices[:N]]

    # Random initialization
    def get_random_init(r):
        pts = []
        while len(pts) < N:
            angle = np.random.rand() * 2 * np.pi
            # Sample radius with sqrt to be uniform in area
            rad = np.sqrt(np.random.rand()) * (1.0 - r)
            pts.append([rad * np.cos(angle), rad * np.sin(angle)])
        return np.array(pts)

    # 2. Binary search on radius r
    # Start with a lower bound from hex packing estimate
    # Max density hex is pi/sqrt(12) ~ 0.9069
    # N * pi * r^2 <= pi * 1^2 * 0.9069  => r^2 <= 0.9069 / 50 => r <= sqrt(0.018) ~ 0.134
    # Known best is 0.1258.
    r_low = 0.120
    r_high = 0.135
    
    # First, try to find a feasible solution for r_low to get a valid baseline
    # If we can't find one, we are in trouble, but hex init should work for 0.12
    
    # We will perform a few restarts at the current best feasible radius
    # and try to push r higher.
    
    current_r = r_low
    current_points = get_hex_init(current_r)
    
    if current_points is None:
        current_points = get_random_init(current_r)
    
    # Optimize current_r
    current_points = _cyclic_coordinate_descent(current_points, current_r, max_iter=500)
    if _is_feasible(current_points, current_r):
        best_score = current_r
        best_points = current_points.copy()
    else:
        # If not feasible, try smaller r
        while r_low > 0.10 and not _is_feasible(current_points, r_low):
             r_low -= 0.001
             current_points = get_hex_init(r_low)
             if current_points is None: break
             current_points = _cyclic_coordinate_descent(current_points, r_low, max_iter=500)
        if _is_feasible(current_points, r_low):
            best_score = r_low
            best_points = current_points.copy()
        else:
            # Fallback to very safe hex
            r_low = 0.10
            current_points = get_hex_init(r_low)
            current_points = _cyclic_coordinate_descent(current_points, r_low, max_iter=1000)
            best_score = _compute_score(current_points)
            best_points = current_points

    # Now, try to increase r
    # Strategy: 
    # 1. Take best_points.
    # 2. Try r_new = current_r * 1.001 (or small step).
    # 3. Run CCD on best_points for r_new.
    # 4. If feasible, update best.
    # 5. If not, try a different initialization (random or perturbed) for r_new.
    
    step = 0.0005
    while time.time() < deadline:
        r_try = best_score + step
        
        # Try to find a feasible solution for r_try
        success = False
        attempts = 0
        max_attempts = 50 # Limit attempts per radius step to avoid getting stuck
        
        while attempts < max_attempts and time.time() < deadline and not success:
            attempts += 1
            
            # Choose initialization
            if attempts == 1:
                # Start from previous best
                init_pts = best_points.copy()
            elif attempts % 2 == 0:
                # Perturbed previous best
                init_pts = best_points + np.random.normal(0, 0.01, size=(N, 2))
                # Ensure inside boundary roughly
                norms = np.linalg.norm(init_pts, axis=1)
                mask = norms > 1.0 - r_try
                init_pts[mask] = init_pts[mask] * (1.0 - r_try) / norms[mask]
            else:
                # Random init
                init_pts = get_random_init(r_try)
            
            # Run CCD
            opt_pts = _cyclic_coordinate_descent(init_pts, r_try, max_iter=1000)
            
            if _is_feasible(opt_pts, r_try):
                success = True
                best_score = r_try
                best_points = opt_pts
                # Reset step? Or keep increasing?
                # Keep increasing.
                # We might be able to jump higher, but small steps are safer.
                # Let's try a slightly larger step if we succeeded quickly?
                # For now, fixed step.
            else:
                # If we used random init and failed, try hex init?
                if attempts == 3:
                    hex_pts = get_hex_init(r_try)
                    if hex_pts is not None:
                        opt_pts = _cyclic_coordinate_descent(hex_pts, r_try, max_iter=1000)
                        if _is_feasible(opt_pts, r_try):
                            success = True
                            best_score = r_try
                            best_points = opt_pts
        
        if not success:
            # If we failed to increase, maybe we are stuck.
            # Try to recover by going back to best_score and doing more optimization?
            # Or reduce step?
            # If we fail at best_score + step, we can't go higher.
            # But maybe our CCD is weak. 
            # Let's try one more time with more iterations on the best points at best_score
            # to see if we can make it "tighter" and allow a jump? 
            # Actually, if it's infeasible at r_try, it's infeasible.
            # We should break if we can't increase.
            # But maybe a different init would work.
            # We did 50 attempts. 
            # Let's break the loop.
            break
            
    # Final check and return
    if best_points is None:
        # Emergency fallback
        best_points = get_hex_init(0.1)
        
    return best_points
# EVOLVE-BLOCK-END
