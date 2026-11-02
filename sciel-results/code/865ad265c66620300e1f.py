"""Rewritten program: gradient-based optimization of sphere covering with 130 points."""
import numpy as np
from scipy.optimize import minimize
import time

N = 130


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius in radians.
    
    The covering radius is the maximum over all points on the sphere of the
    angular distance to the nearest configuration point. This equals the
    maximum circumradius of the facets of the convex hull (for points in
    general position not all in a hemisphere).
    
    We compute it by finding, for each pair of nearby points, the "worst"
    direction - but a more direct approach: the covering radius is the max
    over all points on the sphere. A practical computation: for each point
    on the sphere, find nearest neighbor. The maximum occurs at a point
    equidistant from 3+ points (a Voronoi vertex on the sphere).
    
    For efficiency, we use the fact that the covering radius is the max over
    all pairs of the circumradius of their Delaunay triangle. But computing
    the full Delaunay is expensive. Instead, we use a grid-based approach
    or the fact that for well-spread points, the covering radius is well-
    approximated by the max over all points of the distance to its 3rd
    nearest neighbor... no, that's not right either.
    
    Actually, the cleanest approach: the covering radius is the maximum,
    over all directions u on the sphere, of min_i angular_dist(u, p_i).
    This maximum is achieved at a point equidistant from at least 3 points
    (a vertex of the spherical Voronoi diagram). 
    
    For a practical computation during optimization, we can sample a fine
    grid and take the max. But that's slow. Alternatively, we can use the
    fact that the covering radius is at least the max over all i of
    (1/2) * max_j angular_dist(p_i, p_j) ... no, that's not right.
    
    Let me use a different approach: the covering radius is the radius of
    the smallest cap that covers all points when centered at the "worst"
    point. Equivalently, it's the max over all u of min_i arccos(u·p_i).
    
    For the optimization, I'll use a smooth approximation. But for the
    final evaluation, I need the exact value. Let me use a dense sampling
    plus local refinement.
    """
    # Use a Fibonacci spiral to sample many points on the sphere
    n_samples = 5000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from each sample to all points
    # dots[s, i] = samples[s] · pts[i]
    dots = samples @ pts.T
    # Angular distance = arccos(dots), but we want min over i
    # max over s of min_i arccos(dots[s,i])
    # = arccos(max over s of min_i dots[s,i]) since arccos is decreasing
    min_dots = np.min(dots, axis=1)
    max_min_dot = np.max(min_dots)
    # Clamp for numerical safety
    max_min_dot = np.clip(max_min_dot, -1.0, 1.0)
    return np.arccos(max_min_dot)


def _objective_and_grad(pts):
    """Objective: maximize the minimum pairwise distance (proxy for covering).
    
    Actually, let's directly minimize the covering radius using a smooth
    surrogate. The covering radius R = max_u min_i arccos(u·p_i).
    
    A smooth surrogate: use a soft-max over u and soft-min over i.
    But that's complex. Instead, let's use the fact that minimizing the
    covering radius is related to maximizing the minimum distance between
    points (for equal caps, the optimal covering has the property that
    the covering radius equals the circumradius of each Delaunay triangle).
    
    For a simpler approach: minimize a function that is large when points
    are close together. Use the sum of exp(-k * d_ij) for all pairs, which
    penalizes close points. But this doesn't directly minimize the covering
    radius.
    
    Better approach: use the covering radius itself as the objective, with
    a smooth approximation. The covering radius can be computed as:
    R = max_{u in S^2} min_{i} arccos(u · p_i)
    
    For optimization, I'll use a differentiable proxy: the maximum over
    a set of test points of the soft-min distance.
    """
    # For the objective during optimization, use a smooth approximation
    # of the covering radius. Sample points on the sphere and compute
    # soft-min distances.
    n_samples = 2000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = samples @ pts.T  # shape (n_samples, N)
    # soft-min over i of arccos(dots[s,i]) ≈ -log-sum-exp(-k * arccos(dots))
    # But arccos is not smooth at boundaries. Use a different approach.
    
    # Alternative: minimize the maximum over samples of (1 - min_i dots[s,i])
    # This is a proxy: if min_i dots[s,i] is small, the distance is large.
    # Actually, angular distance = arccos(dot), so minimizing max of arccos
    # is equivalent to maximizing min of dot.
    
    min_dots = np.min(dots, axis=1)
    # Objective: maximize min_dots (i.e., minimize -min_dots)
    # Use soft-max for smoothness: -log-sum-exp(-k * min_dots)
    k = 50.0
    obj = -np.log(np.sum(np.exp(-k * min_dots))) / k
    
    # Gradient: this is complex. Let's use L-BFGS with numerical gradients
    # or just use a simpler objective.
    
    return obj


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Store best configuration and its covering radius
    best_pts = pts.copy()
    best_cr = _covering_radius(best_pts)
    
    # Strategy: iterative local optimization using gradient descent
    # on a smooth surrogate of the covering radius.
    
    # For the surrogate, use the maximum over a set of test directions of
    # the soft-min angular distance to the points.
    
    # Precompute test directions
    n_test = 3000
    t_idx = np.arange(n_test) + 0.5
    tz = 1.0 - 2.0 * t_idx / n_test
    tphi = np.pi * (1.0 + np.sqrt(5.0)) * t_idx
    tr = np.sqrt(1.0 - tz * tz)
    test_pts = np.column_stack([tr * np.cos(tphi), tr * np.sin(tphi), tz])
    
    def smooth_covering_radius(p):
        """Smooth approximation of the covering radius."""
        p = _normalize(p)
        dots = test_pts @ p.T  # (n_test, N)
        # For each test point, compute the angular distance to nearest p_i
        # angular_dist = arccos(dot), but use a smooth min
        # soft-min of arccos values
        # Since arccos is decreasing, soft-min of arccos = arccos(soft-max of dots)
        # soft-max of dots = log-sum-exp(k*dots)/k
        k = 30.0
        max_dots = np.log(np.sum(np.exp(k * dots), axis=1)) / k
        # Clamp
        max_dots = np.clip(max_dots, -1.0, 1.0)
        # Covering radius ≈ max over test points of arccos(max_dots)
        # = arccos(min over test points of max_dots) since arccos is decreasing
        min_max_dot = np.min(max_dots)
        min_max_dot = np.clip(min_max_dot, -1.0, 1.0)
        return np.arccos(min_max_dot)
    
    # Use L-BFGS-B to minimize the smooth covering radius
    # Parameters are the 3N coordinates, but we need to keep them on the sphere.
    # Use a retraction: after each step, normalize.
    
    def objective_flat(x):
        p = x.reshape(N, 3)
        return smooth_covering_radius(p)
    
    # Initial gradient estimate
    x0 = pts.flatten()
    
    # Use Nelder-Mead for robustness, or L-BFGS-B with finite differences
    # Let's try a multi-start approach with local optimization
    
    n_restarts = 8
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            x = x0.copy()
        else:
            # Perturb the current best
            x = best_pts.flatten().copy()
            # Add random perturbation
            noise_scale = 0.1 * (restart / n_restarts)
            x += rng.normal(0, noise_scale, size=(N, 3)).flatten()
        
        # Optimize
        try:
            res = minimize(
                objective_flat,
                x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if res.success or res.fun < smooth_covering_radius(best_pts):
                p_res = _normalize(res.x.reshape(N, 3))
                cr_res = _covering_radius(p_res)
                if cr_res < best_cr:
                    best_cr = cr_res
                    best_pts = p_res.copy()
                    x0 = res.x.copy()
        except Exception:
            pass
    
    # Final polish: use a finer grid for the final covering radius computation
    # and do a few more optimization steps if time permits
    
    if time.time() < deadline - 5.0:
        # Do a final round of optimization with the best configuration
        x = best_pts.flatten()
        try:
            res = minimize(
                objective_flat,
                x,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            p_res = _normalize(res.x.reshape(N, 3))
            cr_res = _covering_radius(p_res)
            if cr_res < best_cr:
                best_cr = cr_res
                best_pts = p_res.copy()
        except Exception:
            pass
    
    return best_pts
