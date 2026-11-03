"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def _compute_radius(points):
    """Compute the maximum radius for the given points."""
    if len(points) == 0:
        return 0.0
    
    # Check container constraint
    dist_to_center = np.linalg.norm(points, axis=1)
    min_container_dist = 1.0 - np.max(dist_to_center)
    
    if min_container_dist < 0:
        # If any point is outside, the radius is effectively negative/invalid
        # But we want to maximize, so we return a very small number or handle in objective
        # For scoring, if invalid, it's 0. But for optimization, we penalize.
        pass

    # Check pairwise distances
    # Use scipy.spatial.distance.pdist for efficiency
    from scipy.spatial.distance import pdist
    d = pdist(points, 'euclidean')
    if len(d) == 0:
        min_pair_dist = float('inf')
    else:
        min_pair_dist = np.min(d)
    
    # The radius is half the minimum distance between centers, 
    # and also limited by the distance to the boundary
    r_pair = min_pair_dist / 2.0
    r_container = min_container_dist
    
    return min(r_pair, r_container)

def _build_initial_rings():
    """
    Initialize points in a ring structure.
    For N=50, a good guess is 1 + 6 + 12 + 18 + 13 or similar.
    Let's try 1 + 6 + 12 + 18 + 13 = 50.
    Radii roughly proportional to sqrt(N) or just heuristic.
    """
    rings = [1, 6, 12, 18, 13]
    # Ensure sum is N
    while sum(rings) < N:
        rings[-1] += 1
    while sum(rings) > N:
        rings[-1] -= 1
        if rings[-1] == 0:
             rings.pop()
             if rings:
                 rings[-1] += 1

    points = []
    # Central circle
    if rings[0] == 1:
        points.append([0.0, 0.0])
    
    current_count = 1 if rings[0] == 1 else 0
    ring_index = 1
    # Heuristic radii
    # r_k ~ k / sqrt(N) * factor
    # Let's estimate based on packing density
    # Area of unit circle = pi. Area of 50 circles = 50 * pi * r^2.
    # r ~ 1/sqrt(50) ~ 0.14. Optimal is 0.125.
    # So r is slightly less.
    
    # Let's just place them evenly spaced in angle and choose initial radii
    # We will optimize these radii.
    # Initial radii guesses: 0, 0.3, 0.5, 0.7, 0.9
    # Let's scale them to fit.
    
    num_rings = len(rings)
    # Distribute radii from 0.1 to 0.9
    radii = np.linspace(0.1, 0.9, num_rings)
    if rings[0] == 1:
         radii = radii[1:] # Skip 0 for center if we handled it, but let's include 0 in params for simplicity?
         # Actually, let's just generate all points including center in the parametric model.
         # Param: for each ring (except center), radius and angle offset.
         
    # Let's define the parameter vector:
    # For each ring i (starting from 1, skipping center if present):
    #   radius_i, angle_offset_i
    # Total params: 2 * (num_rings - 1) if center is fixed at 0.
    # Let's fix center at (0,0) for simplicity, as it's often optimal or near optimal.
    
    params = []
    for i in range(1, num_rings):
        # Initial radius guess
        # Approximate radius for ring i with n_i circles:
        # Circumference 2*pi*R ~ n_i * 2*r => R ~ n_i * r / pi
        # But they are in a ring, so R ~ i * r.
        # Let's just use the linspace values as initial guesses.
        params.append(radii[i-1])
        params.append(0.0) # angle offset

    # Store the structure for reconstruction
    # We need to map params to points.
    # The function _params_to_points will do this.
    
    return params, rings

def _params_to_points(params, rings):
    """Convert parameter vector to list of points."""
    points = []
    # Center
    if rings[0] == 1:
        points.append([0.0, 0.0])
        
    p_idx = 0
    for i in range(1, len(rings)):
        n = rings[i]
        if n == 0:
            continue
            
        r = params[p_idx]
        offset = params[p_idx+1]
        p_idx += 2
        
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + offset
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        pts = np.column_stack((x, y))
        points.extend(pts.tolist())
        
    return np.array(points)

def _objective(params, rings):
    """Objective function to minimize (negative radius)."""
    points = _params_to_points(params, rings)
    
    # Check for NaNs or infinities
    if not np.all(np.isfinite(points)):
        return 1e6
        
    # Check container
    dists = np.linalg.norm(points, axis=1)
    if np.any(dists > 1.0):
        # Penalty for being outside
        penalty = np.sum(dists[dists > 1.0] - 1.0)
        return 100.0 + penalty
        
    # Calculate radius
    r = _compute_radius(points)
    
    # We want to maximize r, so minimize -r
    # Add a small penalty if r is very small to avoid degenerate solutions
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial setup
    # Try a few different ring configurations
    best_points = None
    best_score = -1.0
    
    # Configuration 1: 1 + 6 + 12 + 18 + 13
    rings1 = [1, 6, 12, 18, 13]
    
    # Configuration 2: 1 + 6 + 12 + 15 + 16
    rings2 = [1, 6, 12, 15, 16]
    
    # Configuration 3: 1 + 6 + 12 + 14 + 17
    rings3 = [1, 6, 12, 14, 17]
    
    # Configuration 4: 1 + 6 + 12 + 16 + 15
    rings4 = [1, 6, 12, 16, 15]
    
    configs = [rings1, rings2, rings3, rings4]
    
    for rings in configs:
        if time.time() > deadline:
            break
            
        # Ensure sum is N
        # Adjust last ring if necessary
        diff = N - sum(rings)
        if diff != 0:
            # Adjust the largest ring
            max_idx = np.argmax(rings)
            rings = rings.copy()
            rings[max_idx] += diff
            if rings[max_idx] < 0:
                continue
        
        # Initialize params
        # Number of rings excluding center
        n_rings = len(rings) - 1
        # Initial radii
        # Heuristic: r_i approx i / sqrt(N) * 0.9
        # Let's use a simple linear spacing in radius
        # Max radius is 1-r. Min radius for outer ring is ~0.8
        # Let's guess radii: 0.3, 0.5, 0.7, 0.9
        initial_radii = np.linspace(0.3, 0.9, n_rings)
        
        x0 = []
        for i in range(n_rings):
            x0.append(initial_radii[i])
            x0.append(0.0)
            
        x0 = np.array(x0)
        
        # Bounds
        # Radii must be positive and <= 1.0
        # Angles are unbounded (periodic), but for optimization we can bound them to [0, 2*pi)
        bounds = []
        for i in range(n_rings):
            bounds.append((0.0, 1.0)) # Radius
            bounds.append((0.0, 2.0*np.pi)) # Angle
        
        # Optimize
        try:
            res = opt.minimize(_objective, x0, args=(rings,), method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            
            if res.success or res.fun < -best_score: # res.fun is -r, so lower is better
                points = _params_to_points(res.x, rings)
                score = _compute_radius(points)
                if score > best_score:
                    best_score = score
                    best_points = points
        except Exception:
            pass
            
        # Local refinement: Push apart overlapping circles
        if best_points is not None:
             points = best_points.copy()
             # Simple local search
             for _ in range(50):
                 if time.time() > deadline:
                     break
                 improved = False
                 for i in range(len(points)):
                     for j in range(i+1, len(points)):
                         d = np.linalg.norm(points[i] - points[j])
                         if d < 2*best_score - 1e-9: # Overlap
                             # Push apart
                             diff = points[i] - points[j]
                             if d > 1e-9:
                                 diff = diff / d
                             else:
                                 diff = np.array([1.0, 0.0])
                             # Move both by half the overlap
                             overlap = (2*best_score - d) / 2.0
                             points[i] += diff * overlap
                             points[j] -= diff * overlap
                             improved = True
                             
                 # Check container
                 for i in range(len(points)):
                     d = np.linalg.norm(points[i])
                     if d > 1.0 - best_score:
                         # Push in
                         if d > 1e-9:
                             diff = points[i] / d
                         else:
                             diff = np.array([1.0, 0.0])
                         points[i] -= diff * (d - (1.0 - best_score))
                         improved = True
                         
                 if not improved:
                     break
                     
             # Recalculate score
             new_score = _compute_radius(points)
             if new_score > best_score:
                 best_score = new_score
                 best_points = points
                 
    # Final check
    if best_points is None:
        # Fallback to hexagonal lattice if optimization failed
        # This is the original seed logic, simplified
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_points = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98
            if time.time() > deadline:
                 break

    # Ensure best_points is an array of shape (N, 2)
    if best_points is None or len(best_points) != N:
        # Emergency fallback: random points
        best_points = np.random.rand(N, 2) * 1.8 - 0.9
        # Scale to fit
        d = np.linalg.norm(best_points, axis=1)
        max_d = np.max(d)
        if max_d > 0.8:
            best_points = best_points * (0.8 / max_d)
            
    return best_points

# EVOLVE-BLOCK-END
