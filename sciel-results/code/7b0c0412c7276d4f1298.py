N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, KDTree
from scipy.optimize import minimize
import time


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return pts / norms


def _covering_radius_deg(pts):
    """Fast vectorized covering radius using KDTree on the unit sphere."""
    pts = _normalize(pts)
    # Sample 200k points on the sphere
    rng = np.random.default_rng(42)
    m = 200000
    sample = rng.standard_normal((m, 3))
    sample /= np.linalg.norm(sample, axis=1, keepdims=True)
    tree = KDTree(pts)
    dists, _ = tree.query(sample)
    # dists are chord distances; convert to angular
    # chord = 2*sin(theta/2), so theta = 2*arcsin(chord/2)
    angles = 2.0 * np.arcsin(np.clip(dists / 2.0, 0.0, 1.0))
    return np.degrees(np.max(angles))


def _covering_radius_rad(pts):
    """Covering radius in radians for optimization (same sampling)."""
    pts = _normalize(pts)
    rng = np.random.default_rng(42)
    m = 200000
    sample = rng.standard_normal((m, 3))
    sample /= np.linalg.norm(sample, axis=1, keepdims=True)
    tree = KDTree(pts)
    dists, _ = tree.query(sample)
    angles = 2.0 * np.arcsin(np.clip(dists / 2.0, 0.0, 1.0))
    return np.max(angles)


def _log_sum_exp(x, beta):
    """Soft max approximation."""
    m = np.max(x)
    return m + np.log(np.mean(np.exp(x - m))) / beta


def _covering_objective(x, beta):
    """
    Objective: smooth approximation of covering radius.
    x: flattened 3N parameters (points in R^3, will be normalized).
    """
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    rng = np.random.default_rng(42)
    m = 80000  # fewer samples for speed during optimization
    sample = rng.standard_normal((m, 3))
    sample /= np.linalg.norm(sample, axis=1, keepdims=True)
    tree = KDTree(pts)
    dists, _ = tree.query(sample)
    angles = 2.0 * np.arcsin(np.clip(dists / 2.0, 0.0, 1.0))
    # Use log-sum-exp to get a smooth max
    return _log_sum_exp(angles, beta)


def _covering_objective_grad(x, beta):
    """
    Gradient of the covering objective w.r.t. x (in R^3N space).
    Uses finite differences for simplicity (slow but works).
    Actually, let's use a smarter approach: gradient w.r.t. the normalized points.
    """
    # For speed, we'll use a smaller sample set and analytic-ish gradient
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    rng = np.random.default_rng(42)
    m = 40000
    sample = rng.standard_normal((m, 3))
    sample /= np.linalg.norm(sample, axis=1, keepdims=True)
    tree = KDTree(pts)
    dists, idx = tree.query(sample)
    angles = 2.0 * np.arcsin(np.clip(dists / 2.0, 0.0, 1.0))
    
    # Weight each sample by its softmax weight
    a_max = np.max(angles)
    weights = np.exp(angles - a_max)
    weights /= np.sum(weights)
    
    # Gradient of angle w.r.t. point position:
    # angle = 2*arcsin(d/2), d = ||sample - pt||
    # d(angle)/d(pt) = (1/sqrt(1-(d/2)^2)) * (1/2) * d(d)/d(pt)
    # = (1/sqrt(1-(d/2)^2)) * (pt - sample)/d
    # But we need to account for normalization.
    
    grad = np.zeros_like(x)
    for i in range(N):
        mask = idx == i
        if np.sum(mask) == 0:
            continue
        w = weights[mask]
        s = sample[mask]
        d = dists[mask]
        pt = pts[i]
        # d(angle)/d(pt) for each sample
        # angle = 2*arcsin(d/2)
        # d(angle)/d(d) = 1/sqrt(4-d^2)
        # d(d)/d(pt) = (pt - s)/d
        # But we need to project: the point is constrained to the sphere.
        # Gradient on sphere: project (pt - s)/d * d(angle)/d(d) onto tangent space
        # Then lift back to R^3 gradient (since we normalize, the R^3 gradient is
        # the tangent gradient projected... actually it's simpler to just compute
        # the R^3 gradient of the objective w.r.t. unnormalized x)
        
        # Let's use the chain rule through normalization:
        # If pts = x/||x||, then d(pts)/d(x) = (I - pts*pts^T)/||x||
        # So grad_x = ||x|| * (I - pts*pts^T) * grad_pts
        
        # grad_pts[i] = sum_j w_j * d(angle_j)/d(pts[i])
        # d(angle_j)/d(pts[i]) = (1/sqrt(4-d_j^2)) * (pts[i] - s_j)/d_j
        # But this is the gradient w.r.t. the unnormalized point position.
        # Actually since we normalize, let's just compute the R^3 gradient directly.
        
        # Simpler: just use the formula for gradient of arccos-based angle
        # angle = arccos(dot), where dot = s·pt (for unit vectors)
        # d(angle)/d(pt) = -s / sqrt(1 - dot^2)  (projected to tangent)
        
        dots = np.sum(s * pt, axis=1)
        dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
        # d(angle)/d(pt) = -s / sqrt(1-dot^2)
        denom = np.sqrt(1.0 - dots * dots)
        denom[denom < 1e-12] = 1e-12
        # grad_pts[i] = -sum_j w_j * s_j / denom_j
        grad_pt = -np.sum(w[:, None] * s / denom[:, None], axis=0)
        
        # Project to tangent space
        grad_pt_tangent = grad_pt - np.dot(grad_pt, pt) * pt
        
        # Lift to R^3 gradient (through normalization)
        norm_x = np.linalg.norm(x[i * 3:(i + 1) * 3])
        if norm_x > 1e-14:
            grad[i * 3:(i + 1) * 3] = norm_x * grad_pt_tangent
    
    return grad


def _optimize_from_start(pts, deadline, max_iter=200):
    """Run L-BFGS-B optimization from a starting configuration."""
    x0 = pts.flatten()
    
    # Start with high beta (sharper max), decrease over time
    beta = 200.0
    
    best_x = x0.copy()
    best_val = _covering_objective(x0, beta)
    
    # Use L-BFGS-B with the objective
    # We'll do a few rounds with decreasing beta
    for beta in [300, 200, 150, 100, 80, 60, 50, 40, 35, 30, 25, 20, 15, 12, 10, 8, 6, 5, 4, 3, 2]:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _covering_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'gtol': 1e-4}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
            x0 = res.x
        except Exception:
            pass
    
    return _normalize(best_x.reshape(N, 3))


def _facet_refine(pts, deadline):
    """Refine using convex hull facet analysis (a few iterations)."""
    best_pts = pts.copy()
    best_cr = _covering_radius_deg(pts)
    
    step_size = 0.03
    no_improve = 0
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find facet with largest circumradius
        max_cr = -1.0
        best_simplex = None
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            center_3d = np.mean(facet_pts, axis=0)
            cn = np.linalg.norm(center_3d)
            if cn < 1e-12:
                cr = np.pi / 2
            else:
                center = center_3d / cn
                dots = np.clip(np.sum(facet_pts * center, axis=1), -1.0, 1.0)
                cr = np.max(np.arccos(dots))
            if cr > max_cr:
                max_cr = cr
                best_simplex = simplex
        
        if best_simplex is None:
            break
        
        # Move facet points toward center
        new_pts = pts.copy()
        facet_pts = new_pts[best_simplex]
        center_3d = np.mean(facet_pts, axis=0)
        cn = np.linalg.norm(center_3d)
        if cn < 1e-12:
            break
        center = center_3d / cn
        
        for idx in best_simplex:
            direction = center - new_pts[idx]
            dn = np.linalg.norm(direction)
            if dn > 1e-12:
                new_pts[idx] = new_pts[idx] + step_size * direction / dn
        
        new_pts = _normalize(new_pts)
        new_cr = _covering_radius_deg(new_pts)
        
        if new_cr < best_cr - 1e-8:
            best_pts = new_pts.copy()
            best_cr = new_cr
            pts = new_pts
            no_improve = 0
            step_size = min(step_size * 1.05, 0.08)
        else:
            step_size = max(step_size * 0.85, 0.005)
            no_improve += 1
            if no_improve >= 30:
                # Perturb
                noise = np.random.randn(N, 3) * 0.005
                pts = _normalize(pts + noise)
                no_improve = 0
                step_size = 0.03
    
    return best_pts


def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_pts = _fibonacci_sphere(N)
    best_cr = _covering_radius_deg(best_pts)
    
    # Phase 1: L-BFGS-B optimization from multiple starts
    starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, 1),
        _random_sphere(N, 2),
        _random_sphere(N, 3),
    ]
    
    for i, s in enumerate(starts):
        if time.time() > deadline - 30:  # Leave time for facet refinement
            break
        sub_deadline = min(deadline - 25, start_time + 30 * (i + 1))
        try:
            opt_pts = _optimize_from_start(s, sub_deadline, max_iter=150)
            cr = _covering_radius_deg(opt_pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Facet refinement
    if time.time() < deadline - 5:
        refined = _facet_refine(best_pts, deadline - 2)
        cr = _covering_radius_deg(refined)
        if cr < best_cr:
            best_cr = cr
            best_pts = refined
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
