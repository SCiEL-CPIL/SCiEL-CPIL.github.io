"""Optimized covering of the unit sphere with 130 equal caps."""
import numpy as np
import time
import sys

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees using the convex hull approach.
    The covering radius equals the maximum circumradius of the Delaunay triangles.
    """
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        
        max_circumradius = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v1 = p1 - p2
            v2 = p1 - p3
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            cos_r = np.dot(c, p1)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > max_circumradius:
                max_circumradius = r
        
        return np.degrees(max_circumradius)
    except Exception:
        # Fallback to grid-based computation
        n_theta = 200
        n_phi = 100
        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, np.pi, n_phi)
        
        Theta, Phi = np.meshgrid(theta, phi)
        
        x = np.sin(Phi) * np.cos(Theta)
        y = np.sin(Phi) * np.sin(Theta)
        z = np.cos(Phi)
        
        grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        max_dist = 0.0
        batch_size = 1000
        for i in range(0, len(grid_points), batch_size):
            batch = grid_points[i:i+batch_size]
            dots = batch @ points.T
            cos_dists = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(cos_dists)
            min_dists = np.min(dists, axis=1)
            batch_max = np.max(min_dists)
            if batch_max > max_dist:
                max_dist = batch_max
        
        return np.degrees(max_dist)


def _soft_max_covering_radius(points, temperature=0.05):
    """
    Smooth approximation of the covering radius using soft-max.
    This creates a differentiable objective for L-BFGS-B optimization.
    """
    # Use a moderately fine grid
    n_theta = 100
    n_phi = 50
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi)
    
    Theta, Phi = np.meshgrid(theta, phi)
    
    x = np.sin(Phi) * np.cos(Theta)
    y = np.sin(Phi) * np.sin(Theta)
    z = np.cos(Phi)
    
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Compute angular distances from all grid points to all configuration points
    dots = grid_points @ points.T  # shape (n_grid, N)
    cos_dists = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(cos_dists)  # shape (n_grid, N)
    
    # For each grid point, find the minimum distance (nearest configuration point)
    min_dists = np.min(dists, axis=1)  # shape (n_grid,)
    
    # Use soft-max to approximate the maximum
    scaled = min_dists / temperature
    max_scaled = np.max(scaled)
    soft_max = temperature * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return np.degrees(soft_max)


def _optimize_with_lbfgs(points, max_time=60.0):
    """
    Optimize the covering using L-BFGS-B on a smooth objective.
    Parameters are represented as unconstrained vectors, mapped to the sphere.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    def _params_to_points(params):
        """Convert unconstrained parameters to points on the sphere."""
        # Use a simple mapping: take pairs of parameters as angles
        # Or use a more sophisticated mapping
        # For simplicity, we'll use the fact that we have N points in 3D
        # We'll optimize directly on the sphere using projection
        
        # Actually, let's use a different approach: optimize the points directly
        # but project them back to the sphere after each step
        p = params.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return p / norms
    
    def _objective(params):
        if time.time() - start_time > max_time:
            return 1e6  # Return large value to stop optimization
        p = _params_to_points(params)
        return _soft_max_covering_radius(p, temperature=0.05)
    
    # Initial parameters
    x0 = points.flatten()
    
    # Run L-BFGS-B
    result = minimize(
        _objective,
        x0,
        method='L-BFGS-B',
        options={
            'maxiter': 50,
            'maxfun': 200,
            'ftol': 1e-8,
            'gtol': 1e-6,
            'disp': False
        }
    )
    
    if result.success or result.fun < _soft_max_covering_radius(points, temperature=0.05):
        optimized = _params_to_points(result.x)
        return optimized
    
    return points


def _local_refinement(points, max_time=30.0):
    """
    Local refinement using iterative facet-based adjustments.
    """
    points = points.copy()
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    step_size = 0.02
    decay = 0.98
    
    while time.time() - start_time < max_time:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            
            max_circumradius = 0.0
            worst_simplex = None
            for simplex in hull.simplices:
                p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                v1 = p1 - p2
                v2 = p1 - p3
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                c = n / norm_n
                cos_r = np.dot(c, p1)
                cos_r = np.clip(cos_r, -1.0, 1.0)
                r = np.arccos(cos_r)
                if r > max_circumradius:
                    max_circumradius = r
                    worst_simplex = simplex
            
            if worst_simplex is None:
                break
            
            # Move points in the worst facet away from the circumcenter
            p1, p2, p3 = points[worst_simplex[0]], points[worst_simplex[1]], points[worst_simplex[2]]
            v1 = p1 - p2
            v2 = p1 - p3
            n = np.cross(v1, v2)
            c = n / np.linalg.norm(n)
            
            for idx in worst_simplex:
                # Direction away from circumcenter (tangent to sphere)
                direction = points[idx] - np.dot(points[idx], c) * c
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    points[idx] += step_size * direction
                    points[idx] = points[idx] / np.linalg.norm(points[idx])
            
            # Also try moving a few other points slightly for better balance
            # Find points that are close to the circumcenter
            dists_to_c = np.array([np.arccos(np.clip(np.dot(p, c), -1, 1)) for p in points])
            # Get indices of points closest to circumcenter (excluding those in worst simplex)
            mask = np.ones(N, dtype=bool)
            mask[list(worst_simplex)] = False
            other_indices = np.where(mask)[0]
            if len(other_indices) > 0:
                other_dists = dists_to_c[other_indices]
                # Move the 2-3 closest other points slightly away from circumcenter
                n_move = min(3, len(other_indices))
                closest_others = other_indices[np.argsort(other_dists)[:n_move]]
                for idx in closest_others:
                    direction = points[idx] - np.dot(points[idx], c) * c
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 1e-10:
                        direction = direction / norm_dir
                        points[idx] += 0.5 * step_size * direction
                        points[idx] = points[idx] / np.linalg.norm(points[idx])
            
            # Check if we improved
            current_radius = _covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
            
            step_size *= decay
            if step_size < 1e-5:
                break
                
        except Exception:
            break
    
    return best_points, best_radius


def _fibonacci_spiral():
    """Generate points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def construct():
    """
    Construct 130 points on the unit sphere that minimize the covering radius.
    """
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin before 150s limit
    
    # Generate initial points using Fibonacci spiral
    points = _fibonacci_spiral()
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Phase 1: L-BFGS-B optimization (if time permits)
    remaining = deadline - (time.time() - start_time)
    if remaining > 65.0:
        try:
            lbfgs_points = _optimize_with_lbfgs(points, max_time=min(60.0, remaining - 5.0))
            lbfgs_radius = _covering_radius(lbfgs_points)
            if lbfgs_radius < best_radius:
                best_radius = lbfgs_radius
                best_points = lbfgs_points
        except Exception:
            pass
    
    # Phase 2: Local refinement (if time permits)
    remaining = deadline - (time.time() - start_time)
    if remaining > 35.0:
        try:
            refined_points, refined_radius = _local_refinement(
                best_points, max_time=min(30.0, remaining - 5.0)
            )
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_points = refined_points
        except Exception:
            pass
    
    # Phase 3: Additional L-BFGS-B from the refined result (if time permits)
    remaining = deadline - (time.time() - start_time)
    if remaining > 40.0:
        try:
            final_points = _optimize_with_lbfgs(best_points, max_time=min(35.0, remaining - 5.0))
            final_radius = _covering_radius(final_points)
            if final_radius < best_radius:
                best_radius = final_radius
                best_points = final_points
        except Exception:
            pass
    
    # Ensure valid output
    best_points = np.asarray(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    best_points = best_points / norms
    
    return best_points


# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
