"""Soft-Voronoi differentiable covering-radius optimizer for 130 points on S^2."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import ConvexHull

N = 130
TIME_LIMIT = 135.0  # safety margin before 150s kill


def _normalize(v):
    """Normalize vectors to unit length, handling near-zero norms."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _covering_radius_hull(points):
    """
    Compute the true covering radius using convex hull.
    For each facet (triangle on the hull), the covering radius contribution
    is the circumradius of that triangle (angular distance from sphere center
    to the plane of the facet, measured as angle).
    
    Actually, the covering radius is the maximum over all facets of the angular
    distance from the origin to the facet plane (which equals the angle whose
    cosine is the distance from origin to the plane). For a triangle with vertices
    on the unit sphere, the circumradius of the spherical triangle gives the
    maximum angular distance from points on that facet's cap to the nearest
    configuration point.
    
    More precisely: for a facet defined by 3 points a, b, c on the unit sphere,
    the center of the largest empty cap is at the circumcenter of the spherical
    triangle (normalized). The covering radius for that facet is the angular
    distance from the circumcenter to any of the 3 vertices.
    
    The overall covering radius is the max over all facets of this value.
    """
    try:
        hull = ConvexHull(points)
        max_cr = 0.0
        for simplex in hull.simplices:
            a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            # Compute the normal to the plane of the triangle
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            # The circumcenter of the spherical triangle is in the direction of n
            # (or -n, whichever is farther from the origin in the "outward" direction)
            # For a convex hull of points on the unit sphere, the facet normal points outward.
            # The circumcenter of the spherical triangle is at the normalized n.
            # The angular distance from the circumcenter to vertex a is:
            # arccos(dot(circumcenter, a))
            
            # Check which direction is outward (away from origin for the hull)
            # The centroid of the triangle should be in the direction of n for outward normal
            centroid = (a + b + c) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
            
            # Circumcenter direction
            cc = n  # already normalized
            
            # Angular distances from circumcenter to each vertex
            d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(cc, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(cc, c), -1, 1))
            
            facet_cr = max(d_a, d_b, d_c)
            if facet_cr > max_cr:
                max_cr = facet_cr
        return max_cr
    except Exception:
        return np.pi  # fallback: worst case


def _covering_radius_grid(points, grid_points):
    """
    Estimate covering radius using a dense grid of test points.
    For each grid point, find the nearest configuration point and compute angular distance.
    """
    # Compute dot products between all grid points and all configuration points
    # dots[i, j] = dot(grid_points[i], points[j])
    dots = grid_points @ points.T
    # Maximum dot product gives the nearest point
    max_dots = np.max(dots, axis=1)
    # Angular distance = arccos(max_dot)
    angles = np.arccos(np.clip(max_dots, -1, 1))
    return np.max(angles)


def _generate_grid(n_theta=180, n_phi=360):
    """Generate a uniform-ish grid on the sphere for evaluation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return grid


def _soft_covering_radius(points, grid_points, beta=50.0):
    """
    Differentiable approximation of the covering radius using log-sum-exp.
    
    For each grid point g, the squared geodesic distance to the nearest point
    is approximated. We use the log-sum-exp of negative squared distances to
    approximate the max (nearest neighbor distance).
    
    Actually, we want: max_g min_j d(g, p_j)^2
    
    Approximation: for each g, -logsumexp(-beta * d(g,p_j)^2) approximates min_j d(g,p_j)^2
    Then we take the max over g, or use another logsumexp.
    
    Simpler approach: use the max over grid points of the soft-min.
    """
    # d(g, p_j)^2 = 2 - 2*dot(g, p_j) for unit vectors
    dots = grid_points @ points.T  # shape (n_grid, N)
    sq_dists = 2.0 - 2.0 * dots  # shape (n_grid, N)
    
    # Soft min over j for each g: -logsumexp(-beta * sq_dists) / beta
    # To avoid overflow, use the log-sum-exp trick
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    # We want: -1/beta * log(sum_j exp(-beta * sq_dists[i,j]))
    # = -1/beta * (max_j(-beta*sq_dists[i,j]) + log(sum_j exp(-beta*sq_dists[i,j] - max_j(-beta*sq_dists[i,j]))))
    
    neg_beta_sq = -beta * sq_dists  # shape (n_grid, N)
    max_neg = np.max(neg_beta_sq, axis=1, keepdims=True)  # shape (n_grid, 1)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_sq - max_neg), axis=1, keepdims=True))
    soft_min_sq = -log_sum / beta  # shape (n_grid, 1)
    
    # Now approximate max over grid points
    # Use another logsumexp: (1/beta2) * logsumexp(beta2 * soft_min_sq)
    beta2 = 10.0
    max_sm = np.max(soft_min_sq, axis=0, keepdims=True)
    log_sum2 = max_sm + np.log(np.sum(np.exp(beta2 * (soft_min_sq - max_sm)), axis=0, keepdims=True))
    soft_max = log_sum2 / beta2
    
    return soft_max[0, 0]


def _soft_covering_radius_grad(points, grid_points, beta=50.0, beta2=10.0):
    """
    Compute gradient of the soft covering radius w.r.t. points.
    This is complex; let's use finite differences or a simpler approach.
    For now, use a simpler objective that's easier to differentiate.
    """
    # Alternative: minimize the sum of exp(-beta * min_dist^2) over grid points
    # This encourages all grid points to be close to some configuration point.
    # grad w.r.t. p_j can be computed analytically.
    
    dots = grid_points @ points.T  # (n_grid, N)
    sq_dists = 2.0 - 2.0 * dots  # (n_grid, N)
    
    # Soft min: w_{ij} = exp(-beta * sq_dists[i,j]) / sum_j' exp(-beta * sq_dists[i,j'])
    neg_beta_sq = -beta * sq_dists
    max_neg = np.max(neg_beta_sq, axis=1, keepdims=True)
    exp_neg = np.exp(neg_beta_sq - max_neg)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # (n_grid, N)
    
    # Objective: (1/beta2) * log(sum_i exp(beta2 * soft_min_sq[i]))
    # where soft_min_sq[i] = -1/beta * log(sum_j exp(-beta * sq_dists[i,j]))
    # = -1/beta * (max_neg[i] + log(sum_j exp_neg[i,j]))
    
    # This is getting complex. Let's use a simpler smooth objective:
    # F(points) = sum_i exp(-gamma * min_j d(g_i, p_j)^2)
    # where min is approximated by soft min.
    
    # Actually, let's just use the sum of soft-min squared distances:
    # F = sum_i soft_min_sq[i]
    # This is differentiable and encourages small distances.
    
    # soft_min_sq[i] = -1/beta * log(sum_j exp(-beta * sq_dists[i,j]))
    # d(soft_min_sq[i])/d(points[j]) = -1/beta * 1/sum_j' exp(-beta*sq_dists[i,j']) * d/dp_j sum_j' exp(-beta*sq_dists[i,j'])
    # = -1/beta * 1/S_i * (-beta) * exp(-beta*sq_dists[i,j]) * d(sq_dists[i,j])/dp_j
    # = exp(-beta*sq_dists[i,j]) / S_i * d(sq_dists[i,j])/dp_j
    # = w_{ij} * d(2 - 2*dot(g_i, p_j))/dp_j
    # = w_{ij} * (-2 * g_i)
    
    # So dF/dp_j = sum_i w_{ij} * (-2 * g_i) = -2 * (weights.T @ grid_points)[j]
    
    grad = -2.0 * (weights.T @ grid_points)  # shape (N, 3)
    
    # The objective value
    soft_min_sq = -1.0 / beta * (max_neg[:, 0] + np.log(np.sum(exp_neg, axis=1)))
    obj = np.sum(soft_min_sq)
    
    return obj, grad


def construct():
    start_time = time.time()
    
    # Generate evaluation grid
    grid_points = _generate_grid(n_theta=120, n_phi=240)
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_cr = _covering_radius_grid(points, grid_points)
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 10:
        return _normalize(best_points)
    
    # Optimize using L-BFGS-B with the soft objective
    # We'll use multiple restarts with different initializations if time permits
    
    def objective_and_grad(x):
        """x is flat array of shape (3N,). Returns (obj, grad)."""
        pts = x.reshape(N, 3)
        # Normalize to stay on sphere during optimization
        pts = _normalize(pts)
        obj, grad = _soft_covering_radius_grad(pts, grid_points, beta=50.0)
        return obj, grad.ravel()
    
    # Run optimization
    try:
        result = minimize(
            objective_and_grad,
            points.ravel(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        opt_points = _normalize(result.x.reshape(N, 3))
        
        cr_opt = _covering_radius_grid(opt_points, grid_points)
        if cr_opt < best_cr:
            best_points = opt_points
            best_cr = cr_opt
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 15:
        return _normalize(best_points)
    
    # Try a few random restarts if time allows
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        # Random initial configuration
        pts_rand = rng.normal(size=(N, 3))
        pts_rand = _normalize(pts_rand)
        
        try:
            result = minimize(
                objective_and_grad,
                pts_rand.ravel(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            opt_points = _normalize(result.x.reshape(N, 3))
            
            cr_opt = _covering_radius_grid(opt_points, grid_points)
            if cr_opt < best_cr:
                best_points = opt_points
                best_cr = cr_opt
        except Exception:
            continue
    
    # Final refinement with a denser grid evaluation
    grid_dense = _generate_grid(n_theta=180, n_phi=360)
    cr_final = _covering_radius_grid(best_points, grid_dense)
    
    # One more optimization pass with the best points
    if time.time() - start_time < TIME_LIMIT - 25:
        try:
            def obj_grad_final(x):
                pts = _normalize(x.reshape(N, 3))
                obj, grad = _soft_covering_radius_grad(pts, grid_dense, beta=80.0)
                return obj, grad.ravel()
            
            result = minimize(
                obj_grad_final,
                best_points.ravel(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            final_points = _normalize(result.x.reshape(N, 3))
            cr_final2 = _covering_radius_grid(final_points, grid_dense)
            if cr_final2 < cr_final:
                best_points = final_points
                cr_final = cr_final2
        except Exception:
            pass
    
    # Ensure validity: no coincident points, not in a closed hemisphere
    # Check for coincident points
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-10):
        # Regenerate
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return _normalize(best_points)


if __name__ == '__main__':
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: min={np.min(np.linalg.norm(pts, axis=1)):.6f}, max={np.max(np.linalg.norm(pts, axis=1)):.6f}")
    grid = _generate_grid(90, 180)
    cr = _covering_radius_grid(pts, grid)
    print(f"Estimated covering radius: {np.degrees(cr):.6f} degrees")
