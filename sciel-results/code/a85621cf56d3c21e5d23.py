"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin below 120s

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _compute_covering_radius(pts):
    """
    Compute the covering radius in degrees.
    Uses the convex hull approach: the covering radius is the max circumradius
    of the facets of the convex hull of the points.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        max_circumradius = 0.0
        for simplex in hull.simplices:
            tri = pts[simplex]
            # Compute circumradius of triangle on sphere
            # Using the formula: circumradius = |a| |b| |c| / (4 * area)
            # where a, b, c are side lengths of the triangle
            a = np.linalg.norm(tri[1] - tri[2])
            b = np.linalg.norm(tri[0] - tri[2])
            c = np.linalg.norm(tri[0] - tri[1])
            # Check for degenerate triangle
            if a < 1e-12 or b < 1e-12 or c < 1e-12:
                continue
            # Heron's formula for area
            s = (a + b + c) / 2.0
            area_sq = s * (s - a) * (s - b) * (s - c)
            if area_sq <= 0:
                continue
            area = np.sqrt(area_sq)
            circumradius = a * b * c / (4.0 * area)
            if circumradius > max_circumradius:
                max_circumradius = circumradius
        # Convert to angular distance (radians to degrees)
        # The circumradius of the triangle in 3D corresponds to the angular distance
        # For a triangle on the unit sphere, the circumradius R satisfies:
        # sin(R) = a * b * c / (4 * area) where a,b,c are chord lengths
        # Actually, we need to be more careful. The covering radius is the angular
        # distance from the farthest point to the nearest center.
        # For a facet of the convex hull, the farthest point from the facet's
        # circumcenter is at angular distance R where sin(R) = circumradius_3d
        # But actually, for points on the unit sphere, the angular circumradius
        # is arcsin(circumradius_3d) if circumradius_3d <= 1
        # Wait, let me reconsider. The circumradius of the triangle in 3D space
        # is the distance from the circumcenter to each vertex. Since the vertices
        # are on the unit sphere, the circumcenter is at some point, and the
        # circumradius R_3d satisfies R_3d^2 = 1 - h^2 where h is the distance
        # from the origin to the circumcenter plane.
        # The angular distance from the circumcenter (projected to sphere) to a
        # vertex is arccos(h) = arcsin(R_3d) if R_3d <= 1.
        # Actually, for a triangle on the unit sphere, the angular circumradius
        # theta satisfies: sin(theta) = R_3d where R_3d is the 3D circumradius.
        # So theta = arcsin(R_3d).
        if max_circumradius > 1.0:
            max_circumradius = 1.0
        angular_radius = np.arcsin(max_circumradius)
        return np.degrees(angular_radius)
    except Exception:
        return 180.0  # Fallback: worst case

def _soft_max_covering_radius(pts, beta=100.0):
    """
    Soft approximation of the covering radius using log-sum-exp.
    This is differentiable and can be used with gradient-based optimizers.
    """
    pts = _normalize(pts)
    # Sample points on the sphere to approximate the max distance
    # Use a quasi-random sampling for efficiency
    n_samples = 2000
    # Fibonacci sphere for sampling
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distance from each sample to each point
    # dot products
    dots = samples @ pts.T  # shape (n_samples, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # angular distances in radians
    
    # Soft max over the nearest point for each sample
    # For each sample, find the min angle to any point
    min_angles = np.min(angles, axis=1)  # shape (n_samples,)
    
    # Soft max of min_angles
    beta_scaled = beta
    max_val = np.max(min_angles)
    soft_max = max_val + np.log(np.sum(np.exp(beta_scaled * (min_angles - max_val)))) / beta_scaled
    
    return np.degrees(soft_max)

def _objective_and_grad(pts_flat, beta=100.0):
    """
    Objective function and gradient for L-BFGS-B.
    Uses finite differences for gradient since analytical is complex.
    """
    pts = pts_flat.reshape(-1, 3)
    val = _soft_max_covering_radius(pts, beta=beta)
    # Use numerical gradient
    eps = 1e-6
    grad = np.zeros_like(pts_flat)
    for i in range(len(pts_flat)):
        pts_plus = pts_flat.copy()
        pts_plus[i] += eps
        val_plus = _soft_max_covering_radius(pts_plus.reshape(-1, 3), beta=beta)
        grad[i] = (val_plus - val) / eps
    return val, grad

def _optimize_from_start(start_pts, beta_schedule=None, max_time=None):
    """
    Optimize from a starting configuration using L-BFGS-B with beta cascade.
    """
    if beta_schedule is None:
        beta_schedule = [15, 50, 150, 300, 500]
    if max_time is None:
        max_time = TIME_LIMIT - (time.time() - START_TIME) - 5.0  # 5s safety margin
    
    pts = _normalize(start_pts.copy())
    best_val = _compute_covering_radius(pts)
    
    for beta in beta_schedule:
        if time.time() - START_TIME > max_time:
            break
        # Flatten for L-BFGS-B
        pts_flat = pts.flatten()
        
        def obj(x):
            return _soft_max_covering_radius(x.reshape(-1, 3), beta=beta)
        
        result = minimize(
            obj,
            pts_flat,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_val:
            new_pts = _normalize(result.x.reshape(-1, 3))
            new_val = _compute_covering_radius(new_pts)
            if new_val < best_val:
                pts = new_pts
                best_val = new_val
            else:
                # Keep old if not better
                pass
        else:
            # Try to salvage
            new_pts = _normalize(result.x.reshape(-1, 3))
            new_val = _compute_covering_radius(new_pts)
            if new_val < best_val:
                pts = new_pts
                best_val = new_val
    
    return pts, best_val

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    global START_TIME
    START_TIME = time.time()
    
    # Generate diverse initial configurations
    starts = []
    
    # 1. Golden-angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    starts.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random with fixed seed
    np.random.seed(42)
    for seed in [1, 2, 3, 4, 5]:
        np.random.seed(seed)
        pts = np.random.randn(N, 3)
        pts = _normalize(pts)
        starts.append(pts)
    
    # 3. Perturbed golden spiral
    np.random.seed(7)
    base = starts[0].copy()
    noise = np.random.randn(N, 3) * 0.1
    perturbed = _normalize(base + noise)
    starts.append(perturbed)
    
    # 4. Another perturbation
    np.random.seed(13)
    noise = np.random.randn(N, 3) * 0.3
    perturbed2 = _normalize(base + noise)
    starts.append(perturbed2)
    
    # 5. Uniform random on sphere using a different method
    np.random.seed(99)
    u = np.random.uniform(-1, 1, N)
    v = np.random.uniform(-1, 1, N)
    w = np.random.uniform(-1, 1, N)
    pts = np.column_stack([u, v, w])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    starts.append(pts)
    
    # 6. More random
    np.random.seed(123)
    pts = np.random.randn(N, 3)
    pts = _normalize(pts)
    starts.append(pts)
    
    # 7. Yet another
    np.random.seed(456)
    pts = np.random.randn(N, 3)
    pts = _normalize(pts)
    starts.append(pts)
    
    # Optimize from each start, keep the best
    best_pts = starts[0]
    best_val = _compute_covering_radius(best_pts)
    
    # Allocate time per start
    n_starts = len(starts)
    time_per_start = max(5.0, (TIME_LIMIT - (time.time() - START_TIME) - 10.0) / n_starts)
    
    for idx, start in enumerate(starts):
        remaining = TIME_LIMIT - (time.time() - START_TIME) - 5.0
        if remaining < 10.0:
            break
        
        max_time_for_this = min(time_per_start, remaining - 5.0)
        if max_time_for_this < 5.0:
            break
        
        try:
            pts, val = _optimize_from_start(start, max_time=max_time_for_this)
            if val < best_val:
                best_val = val
                best_pts = pts
        except Exception:
            continue
    
    # Final polish with tighter tolerances if time allows
    remaining = TIME_LIMIT - (time.time() - START_TIME) - 3.0
    if remaining > 20.0:
        try:
            pts, val = _optimize_from_start(
                best_pts,
                beta_schedule=[300, 500, 800],
                max_time=remaining - 5.0
            )
            if val < best_val:
                best_val = val
                best_pts = pts
        except Exception:
            pass
    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
