import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Fast calculation of minimum triangle area using vectorized operations.
    Points: (N, 2) array. Returns minimum area (not doubled).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            v2s = points[j + 1:] - pi
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                cur_min = np.min(areas)
                if cur_min < min_area:
                    min_area = cur_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _negative_score(points):
    """
    Objective function for minimizer.
    """
    return -_score(points)

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _cyclic_symmetric_points(radii, offsets):
    """
    Construct points with cyclic symmetry.
    radii: array of length 7 (radii for the first 7 points in the symmetric sector)
    offsets: array of length 7 (angular offsets from the ideal regular polygon angles)
    
    The points are constructed as:
    P_k = r_k * exp(i * (2*pi*k/14 + offset_k)) for k=0..6
    P_{k+7} = r_k * exp(i * (2*pi*(k+7)/14 + offset_k)) = r_k * exp(i * (2*pi*k/14 + pi + offset_k))
    
    Wait, strict cyclic symmetry of order 14 would mean P_{k+7} is a rotation of P_k by pi.
    However, the optimal configuration for even N often has "dimer" structures or deviations.
    Let's allow independent radii and angles for all 14 points but enforce a specific structure:
    We assume the optimal solution is close to a regular polygon.
    Let's parameterize by 14 radii and 14 angles relative to the regular polygon.
    Actually, to keep dimensionality low and exploit symmetry:
    Let's assume the set of points is invariant under rotation by 2*pi/14.
    This implies all radii are equal and all offsets are equal. That's just a regular polygon.
    
    Let's relax to: The points come in pairs? Or just use a full parameterization but optimize with a smarter method?
    
    Alternative Strategy: 
    The "different strategy" is to use **Gradient Descent with Analytic Gradients** or **Nelder-Mead on a reduced parameter space**.
    
    Let's try a reduced parameter space based on the assumption that the optimal configuration for n=14 
    is a perturbation of the regular polygon where we adjust radii and angles independently but smoothly.
    
    Actually, a very effective strategy for these "max-min" problems is **Simulated Annealing** or **L-BFGS-B** 
    on a parameterization that guarantees convex position.
    
    Let's stick to a **Coordinate Ascent** method on the 28 coordinates, but with a **smart initialization** 
    and **adaptive step sizes**. This is different from the global CMA-ES approach.
    
    Even better: **Gradient-based optimization** using finite differences or analytic gradients.
    The score function is min(area_i) / hull_area. It's non-smooth (min), but we can smooth it or use subgradients.
    
    Let's implement a **Subgradient Method** / **Projected Gradient Descent**.
    We want to maximize min(area_i). 
    Let A_i be the area of triangle i.
    We want to maximize t such that A_i >= t for all i.
    This is a semi-infinite program, but we can use the "max-min" reformulation.
    
    Objective: f(x) = min_i (A_i(x) / H(x))
    
    Let's use **Nelder-Mead** but on a **log-polar parameterization** to ensure positivity of radii and handle angles naturally, 
    and initialize from a **distorted regular polygon** that has been pre-optimized by a few steps of a fast heuristic.
    
    Actually, the current program already uses Nelder-Mead. The key difference is **Initialization** and **Parameterization**.
    
    New Strategy: **Random Restarts with Nelder-Mead on a Log-Polar Parameter Space**.
    The current program uses Cartesian coordinates. Log-polar (r, theta) is often better for radial problems.
    Also, we will perform **many more restarts** with smaller step sizes, leveraging the speed of the vectorized score.
    
    Let's implement a **Multi-start Nelder-Mead** with **Log-Polar coordinates**.
    """
    pass # Placeholder for logic inside construct

def _to_polar(points):
    """Convert Cartesian to (log_r, theta)."""
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    theta = np.arctan2(points[:, 1], points[:, 0])
    # Handle r=0
    r = np.where(r < 1e-10, 1e-10, r)
    return np.column_stack([np.log(r), theta])

def _from_polar(log_r_theta):
    """Convert (log_r, theta) to Cartesian."""
    r = np.exp(log_r_theta[:, 0])
    theta = log_r_theta[:, 1]
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def _score_polar(log_r_theta):
    """Score function in log-polar space."""
    points = _from_polar(log_r_theta)
    return _score(points)

def _negative_score_polar(log_r_theta):
    return -_score_polar(log_r_theta)

def _nelder_mead_polar(x0, maxiter=100, deadline=None):
    """Nelder-Mead in log-polar space."""
    from scipy.optimize import minimize
    
    def callback(xk):
        if deadline is not None and time.time() > deadline:
            raise KeyboardInterrupt()
    
    try:
        res = minimize(_negative_score_polar, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
                       callback=callback)
        return res.x, -res.fun
    except KeyboardInterrupt:
        return x0, -_negative_score_polar(x0)
    except Exception:
        return x0, -_negative_score_polar(x0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_points = points.copy()
            best_score = score
    
    # Initialization: Regular polygon in log-polar space
    base_points = _regular_polygon(N)
    base_polar = _to_polar(base_points)
    
    # Strategy: Multi-start Nelder-Mead in Log-Polar space
    # We will try many random perturbations of the regular polygon in log-polar space.
    
    rng = np.random.default_rng(42)
    
    # Number of restarts depends on time budget
    # Each restart takes ~50-100ms with maxiter=100
    # We have ~145s. Let's aim for 200-500 restarts if fast, or fewer with more iterations.
    
    # First, do a quick local search from the perfect regular polygon
    try:
        opt_polar, opt_score = _nelder_mead_polar(base_polar, maxiter=200, deadline=deadline - 10)
        opt_points = _from_polar(opt_polar)
        update_best(opt_points, opt_score)
    except Exception:
        pass
    
    # Now, multi-start with perturbations
    # Perturb log_r and theta
    # log_r perturbations: small, since scale is normalized by hull area, but relative radii matter
    # theta perturbations: break symmetry
    
    max_restarts = 300
    restart_count = 0
    
    while time.time() < deadline - 5:
        restart_count += 1
        
        # Generate perturbation
        # Scale of perturbation decreases as we go? Or keep it moderate to escape local optima.
        # Let's use a schedule: start with larger jumps, end with smaller.
        # Or just random uniform in a box.
        
        # Perturb theta: uniform in [-0.2, 0.2]
        # Perturb log_r: uniform in [-0.1, 0.1]
        
        perturbation = np.zeros_like(base_polar)
        perturbation[:, 0] = rng.uniform(-0.1, 0.1, N) # log_r
        perturbation[:, 1] = rng.uniform(-0.2, 0.2, N) # theta
        
        x0 = base_polar + perturbation
        
        try:
            opt_polar, opt_score = _nelder_mead_polar(x0, maxiter=100, deadline=deadline - 2)
            opt_points = _from_polar(opt_polar)
            update_best(opt_points, opt_score)
        except Exception:
            pass
        
        # If we found a significantly better score, we might want to refine it further
        if opt_score > best_score - 1e-6:
             # Do a quick refinement on the best found so far
             if time.time() < deadline - 5:
                 try:
                     # Refine the best polar coordinates found so far
                     best_polar = _to_polar(best_points)
                     opt_polar, opt_score = _nelder_mead_polar(best_polar, maxiter=200, deadline=deadline - 2)
                     opt_points = _from_polar(opt_polar)
                     update_best(opt_points, opt_score)
                 except Exception:
                     pass

    if best_points is None:
        best_points = _regular_polygon(N)
    
    return best_points

# EVOLVE-BLOCK-END
