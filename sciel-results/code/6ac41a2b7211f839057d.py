"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _initial_fib():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_data(pts):
    """
    Compute the circumcenters and circumradii (cosines) of the convex hull facets.
    Returns:
        facets: list of simplices
        cc_dirs: array of unit vectors pointing to circumcenters
        cos_radii: array of cosines of the circumradii (distance from origin to facet plane)
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return [], np.array([]), np.array([])
    
    facets = hull.simplices
    cc_dirs = []
    cos_radii = []
    
    for simplex in facets:
        v1 = pts[simplex[0]]
        v2 = pts[simplex[1]]
        v3 = pts[simplex[2]]
        
        e1 = v2 - v1
        e2 = v3 - v1
        n = np.cross(e1, e2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-12:
            continue
        
        # The normal vector n points outward from the convex hull (assuming consistent orientation)
        # The distance from origin to the plane is d = dot(n, v1) / n_norm
        # Since points are on unit sphere and origin is inside, d should be positive for outward normals.
        # However, scipy's hull normals might be oriented arbitrarily relative to origin? 
        # Actually, for a convex body containing the origin, the support function h(n) = max dot(x, n) is the distance.
        # The plane equation is n . x = d. Since origin is inside, 0 < d < 1.
        
        d = np.dot(n, v1) / n_norm
        
        # Ensure d is positive (outward normal)
        if d < 0:
            d = -d
            n = -n
            
        # Normalize n to get the direction of the circumcenter (which is the farthest point in that direction)
        cc_dir = n / n_norm
        
        # The circumradius is the angle of the circumcenter.
        # cos(angle) = d
        cc_dirs.append(cc_dir)
        cos_radii.append(d)
        
    return facets, np.array(cc_dirs), np.array(cos_radii)

def _covering_radius(pts):
    """Compute the exact covering radius in degrees using convex hull facets."""
    if len(pts) < 4:
        return 180.0
    
    facets, cc_dirs, cos_radii = _compute_hull_data(pts)
    if len(cos_radii) == 0:
        return 180.0
        
    # The covering radius is the maximum angle among the circumcenters.
    # angle = arccos(cos_radius)
    # We want to minimize this, so we maximize the minimum cos_radius.
    min_cos = np.min(cos_radii)
    min_cos = np.clip(min_cos, -1.0, 1.0)
    
    return np.degrees(np.arccos(min_cos))

def _least_squares_polish(pts, target_cos, max_iter=50, tol=1e-8):
    """
    Use least_squares to minimize the violation of the constraint:
    cos(angle_i) >= target_cos for all facets i.
    
    Residuals: For each facet, if cos_i < target_cos, residual = target_cos - cos_i.
               Otherwise, residual = 0.
    To make it smooth, we can use a soft hinge or just minimize the square of the negative part.
    scipy.optimize.least_squares handles non-smoothness somewhat via finite differences or provided jacobians,
    but it's better to provide a smooth approximation or use the 'trf' method which handles bounds well.
    
    Actually, a standard trick for "minimize max violation" is to minimize the sum of squared violations.
    Let v_i = max(0, target_cos - cos_i). Minimize sum(v_i^2).
    This is non-smooth at v_i=0.
    
    Alternative: Minimize sum((target_cos - cos_i)^2) subject to cos_i <= 1? No, that pulls everything to target.
    
    Let's use a differentiable surrogate for the max constraint violation.
    Or, simply minimize the sum of squares of the "shortfalls".
    Shortfall_i = target_cos - cos_i.
    If Shortfall_i > 0, we have a violation.
    
    Let's define the objective for least_squares as:
    r_i = target_cos - cos_i(pts)
    We want r_i <= 0 for all i.
    Minimizing sum(r_i^2) will try to make all r_i close to 0, which means all cos_i close to target_cos.
    This is a valid approach: "Make all facets have circumradius exactly R_target".
    If the system is feasible, this will find a solution where all are >= R_target (or close).
    If infeasible, it will find the point that minimizes the deviation from the target, effectively finding the best possible radius near the target.
    
    However, if we just minimize sum(r_i^2), we might shrink the radius too much if it's easy to do so?
    No, because we are iterating target_cos downwards.
    Start with a loose target (e.g., cos(15 deg)).
    Optimize to make all facets have cos >= target.
    Then decrease target slightly.
    Optimize again.
    
    The objective for least_squares:
    residuals = target_cos - cos_radii(pts)
    We minimize sum(residuals^2).
    This drives cos_radii towards target_cos.
    If current cos_radii > target_cos, the residual is negative, squared is positive.
    The gradient will push to increase cos_radii (decrease angle) until it hits target?
    Wait. If cos_i > target, residual is negative. Gradient of r^2 is 2r dr/dx.
    dr/dx = - d(cos_i)/dx.
    So grad = -2 * (target - cos_i) * d(cos_i)/dx.
    If cos_i > target, (target - cos_i) is negative.
    So grad = -2 * (neg) * d(cos_i)/dx = +2 * (cos_i - target) * d(cos_i)/dx.
    This pushes x in the direction that increases cos_i further?
    Yes, it pushes away from the target if already above it.
    This is bad. We want to keep them >= target.
    
    Correct approach for constraints:
    We want to minimize the "badness" of violations.
    Let v_i = max(0, target_cos - cos_i).
    We want to minimize sum(v_i^2).
    This is non-differentiable at v_i=0.
    
    Smooth approximation:
    v_i_smooth = 0.5 * (v_i + sqrt(v_i^2 + eps))
    Or simply use `scipy.optimize.minimize` with method 'SLSQP' and constraints?
    SLSQP is slow for 390 variables.
    
    Let's go back to `least_squares`.
    We can define residuals such that they are zero when satisfied, and positive when violated.
    But least_squares minimizes sum(r^2).
    If we define r_i = target_cos - cos_i, and we only care about r_i > 0,
    we can use a "soft" version:
    r_i = target_cos - cos_i
    We want to minimize sum( max(0, r_i)^2 ).
    
    Since least_squares doesn't support max directly in a smooth way,
    let's use a differentiable function:
    f(r) = r^2 if r > 0, else 0.
    This is C1 continuous. Gradient is 2r if r>0, else 0.
    
    So, residual for facet i:
    r_i = target_cos - cos_i
    If r_i > 0, residual = r_i.
    If r_i <= 0, residual = 0.
    
    We need to provide the Jacobian.
    d(residual)/d(x) = d(r_i)/d(x) if r_i > 0 else 0.
    d(r_i)/d(x) = - d(cos_i)/d(x).
    
    d(cos_i)/d(x):
    cos_i = dot(n_i, x_{i0}) / ||n_i|| where n_i = cross(x_{i1}-x_{i0}, x_{i2}-x_{i0}).
    This is complex.
    
    Alternative: Use finite difference Jacobian. It's 390 vars.
    Each function eval is O(N^2) or O(Hull).
    Hull is expensive.
    
    Let's stick to a simpler strategy that is known to work for this specific problem type in evolutionary contexts:
    **Iterative Facet Pushing with Line Search.**
    
    1. Compute Hull.
    2. Identify the facet with the largest circumradius (smallest cosine).
    3. Move the 3 vertices of that facet slightly towards the circumcenter (which reduces the circumradius of that facet, but might increase others).
    4. Use a line search to find the step size that minimizes the *maximum* circumradius.
    
    This is effectively a subgradient method on the max function.
    """
    current_pts = pts.copy()
    current_radius = _covering_radius(current_pts)
    
    for it in range(max_iter):
        facets, cc_dirs, cos_radii = _compute_hull_data(current_pts)
        if len(cos_radii) == 0:
            break
            
        # Find the worst facet (min cosine = max angle)
        worst_idx = np.argmin(cos_radii)
        worst_cos = cos_radii[worst_idx]
        
        # If we already meet the target, we might stop or refine
        if worst_cos >= target_cos:
             # We could break, but maybe we can tighten further?
             # For this polish step, if we meet target, we are good for this iteration.
             # But we want to minimize the actual radius, not just meet target.
             # So we continue to push even if target is met, to lower the radius.
             pass
             
        simplex = facets[worst_idx]
        cc_dir = cc_dirs[worst_idx]
        
        # Direction to move: towards the circumcenter
        # The circumcenter is at cc_dir (on the sphere).
        # Moving vertices towards cc_dir reduces the distance to the origin for the plane?
        # Yes, pushing vertices "out" towards the hole center reduces the circumradius.
        
        # Try a small step
        best_step = 0.0
        best_new_radius = current_radius
        
        # Simple line search: try a few step sizes
        for step in [0.01, 0.005, 0.002, 0.001]:
            new_pts = current_pts.copy()
            for idx in simplex:
                # Move towards cc_dir
                new_pts[idx] = new_pts[idx] + step * cc_dir
                # Normalize
                new_pts[idx] /= np.linalg.norm(new_pts[idx])
            
            new_radius = _covering_radius(new_pts)
            
            if new_radius < best_new_radius:
                best_new_radius = new_radius
                best_step = step
                best_pts = new_pts.copy()
                
        if best_step == 0.0:
            break
            
        current_pts = best_pts
        current_radius = best_new_radius
        
        # Check convergence
        if it > 0 and abs(prev_radius - current_radius) < tol:
            break
        prev_radius = current_radius
        
    return current_pts

def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    max_time = 145.0  # Safety margin
    
    # Start with Fibonacci spiral
    pts = _initial_fib()
    
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # Phase 1: Coarse Refinement via Facet Pushing
    # We don't have a fixed target yet. We just push the worst facet.
    # This is effectively minimizing the max circumradius.
    
    current = best_pts.copy()
    current_radius = best_radius
    
    # Iterate until time runs out or convergence
    # We use a decreasing step size or just fixed small steps with line search.
    
    for it in range(200):  # Max iterations
        if time.time() - start_time > max_time - 10:
            break
            
        facets, cc_dirs, cos_radii = _compute_hull_data(current)
        if len(cos_radii) == 0:
            break
            
        worst_idx = np.argmin(cos_radii)
        worst_cos = cos_radii[worst_idx]
        simplex = facets[worst_idx]
        cc_dir = cc_dirs[worst_idx]
        
        # Line search for step size
        best_step = 0.0
        best_new_radius = current_radius
        best_candidate = current.copy()
        
        # Adaptive step sizes based on current radius?
        # Just use a fixed set for now.
        steps = [0.05, 0.02, 0.01, 0.005, 0.002]
        
        for step in steps:
            new_pts = current.copy()
            for idx in simplex:
                new_pts[idx] = new_pts[idx] + step * cc_dir
                new_pts[idx] /= np.linalg.norm(new_pts[idx])
            
            # Quick check: is it still a valid hull?
            # _covering_radius handles exceptions.
            new_radius = _covering_radius(new_pts)
            
            if new_radius < best_new_radius:
                best_new_radius = new_radius
                best_step = step
                best_candidate = new_pts.copy()
                
        if best_step == 0.0:
            # Try smaller steps if large ones failed
            small_steps = [0.001, 0.0005, 0.0002]
            for step in small_steps:
                new_pts = current.copy()
                for idx in simplex:
                    new_pts[idx] = new_pts[idx] + step * cc_dir
                    new_pts[idx] /= np.linalg.norm(new_pts[idx])
                new_radius = _covering_radius(new_pts)
                if new_radius < best_new_radius:
                    best_new_radius = new_radius
                    best_step = step
                    best_candidate = new_pts.copy()
                    
        if best_step == 0.0:
            # Stuck? Try a random perturbation to escape local minimum
            # Perturb the worst facet points slightly randomly
            current = best_candidate
            # Add small noise
            noise = np.random.randn(N, 3) * 0.01
            current = current + noise
            current /= np.linalg.norm(current, axis=1, keepdims=True)
            current_radius = _covering_radius(current)
            if current_radius > best_radius:
                # Revert if it got worse
                current = best_pts.copy()
                current_radius = best_radius
            continue
            
        current = best_candidate
        current_radius = best_new_radius
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current.copy()
            
        # Convergence check
        if it > 0:
             # If improvement is tiny, we might be converging
             # We don't have prev_radius stored in outer scope easily, 
             # but we can check if best_radius hasn't improved in last K iterations?
             # For now, just run until time limit.
             pass

    # Final polish with a very small step size
    if time.time() - start_time < max_time - 2:
        current = best_pts.copy()
        for it in range(50):
            if time.time() - start_time > max_time - 1:
                break
            facets, cc_dirs, cos_radii = _compute_hull_data(current)
            if len(cos_radii) == 0: break
            worst_idx = np.argmin(cos_radii)
            simplex = facets[worst_idx]
            cc_dir = cc_dirs[worst_idx]
            
            improved = False
            for step in [0.001, 0.0005, 0.0002]:
                new_pts = current.copy()
                for idx in simplex:
                    new_pts[idx] = new_pts[idx] + step * cc_dir
                    new_pts[idx] /= np.linalg.norm(new_pts[idx])
                new_radius = _covering_radius(new_pts)
                if new_radius < _covering_radius(current):
                    current = new_pts
                    improved = True
                    break
            if not improved:
                break
        
        final_radius = _covering_radius(current)
        if final_radius < best_radius:
            best_pts = current
            best_radius = final_radius

    # Ensure no coincident points
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-10:
                best_pts[j] += np.random.randn(3) * 0.001
                best_pts[j] /= np.linalg.norm(best_pts[j])
                
    return best_pts
# EVOLVE-BLOCK-END
