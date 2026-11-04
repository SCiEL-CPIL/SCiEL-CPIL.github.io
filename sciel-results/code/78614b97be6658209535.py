"""Combined approach: multi-start + smooth LSE covering-radius refinement + Riesz polish."""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# ---------------- utilities ----------------

def _normalize(points):
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _covering_radius(points):
    """Exact covering radius (radians) via convex hull facet circumcenters."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: dense random sampling
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 20000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))

    max_r = 0.0
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r

    return max_r


def _facet_data(points):
    """Return list of (radius, simplex_indices, circumcenter) for each hull facet."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    facets = []
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        facets.append((r, simplex, v))
    return facets


def _smooth_covering_radius_grad(x, beta):
    """
    Smooth LSE approximation of covering radius.
    x: flat array of shape (3N,)
    Returns (value, gradient).
    """
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    pts = pts / norms
    
    facets = _facet_data(pts)
    if not facets:
        return 1.0, np.zeros_like(x)
    
    radii = np.array([f[0] for f in facets])
    # LSE: (1/beta) * log(sum(exp(beta * r_i)))
    max_r = np.max(radii)
    shifted = beta * (radii - max_r)
    # Clip for numerical stability
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted)))
    value = max_r + log_sum / beta
    
    # Gradient: d/dx [ (1/beta) log sum exp(beta r_i) ]
    # = sum_i (exp(beta r_i) / sum exp(beta r_j)) * dr_i/dx
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad_pts = np.zeros_like(pts)
    for w, (r, simplex, center) in zip(weights, facets):
        if w < 1e-15:
            continue
        for idx in simplex:
            p = pts[idx]
            # dr/dp: derivative of arccos(dot(v, a)) w.r.t. a
            # v depends on a,b,c but for small perturbations, treat v as fixed
            # dr/dp_i = -v / sqrt(1 - dot(v,p_i)^2) projected to tangent
            dot_vp = np.dot(center, p)
            dot_vp = np.clip(dot_vp, -1 + 1e-10, 1 - 1e-10)
            denom = np.sqrt(1.0 - dot_vp * dot_vp)
            if denom < 1e-10:
                continue
            d_r_dp = -center / denom
            # Project to tangent plane
            d_r_dp = d_r_dp - np.dot(d_r_dp, p) * p
            grad_pts[idx] += w * d_r_dp
    
    grad_x = grad_pts.reshape(-1)
    return value, grad_x


def _riesz_energy_grad(x, s=2.0):
    """
    Riesz s-energy: sum_{i!=j} 1/d(i,j)^s
    Minimize to spread points. Returns (value, gradient).
    """
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    pts = pts / norms
    
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, 1.0)  # avoid division by zero
    
    # Energy
    inv_dists = 1.0 / np.maximum(dists, 1e-10)
    energy = np.sum(inv_dists ** s) - N  # exclude self terms
    
    # Gradient
    grad_pts = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-10:
                continue
            # d/dp_i [1/d^s] = -s * (p_i - p_j) / d^(s+2)
            vec = pts[i] - pts[j]
            grad_pts[i] += -s * vec / (d ** (s + 2))
    
    # Project to tangent
    for i in range(N):
        p = pts[i]
        grad_pts[i] = grad_pts[i] - np.dot(grad_pts[i], p) * p
    
    grad_x = grad_pts.reshape(-1)
    return energy, grad_x


def _lbfgs_smooth(points, beta, time_limit=None, start_time=None, maxiter=300):
    """L-BFGS on smooth LSE covering radius."""
    x0 = points.flatten()
    
    def obj_and_grad(x):
        return _smooth_covering_radius_grad(x, beta)
    
    result = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'gtol': 1e-10, 'maxfun': 800, 'maxls': 50})
    
    x_opt = result.x.reshape(N, 3)
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    cr = _covering_radius(x_opt)
    return x_opt, cr


def _lbfgs_riesz(points, s=2.0, time_limit=None, start_time=None, maxiter=150):
    """L-BFGS on Riesz energy for diversification."""
    x0 = points.flatten()
    
    def obj_and_grad(x):
        return _riesz_energy_grad(x, s)
    
    result = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'gtol': 1e-8, 'maxfun': 400})
    
    x_opt = result.x.reshape(N, 3)
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    cr = _covering_radius(x_opt)
    return x_opt, cr


def _lloyd_iteration(points, max_iters=300, tol=1e-12, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        facets = _facet_data(points)
        if not facets:
            break
        
        new_points = np.zeros_like(points)
        point_to_facets = [[] for _ in range(N)]
        for fi, (r, simplex, center) in enumerate(facets):
            for idx in simplex:
                point_to_facets[idx].append(fi)
        
        for i in range(N):
            if not point_to_facets[i]:
                new_points[i] = points[i]
                continue
            
            circumcenters = [facets[fi][2] for fi in point_to_facets[i]]
            centroid = np.mean(circumcenters, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                new_points[i] = points[i]
            else:
                new_points[i] = centroid / centroid_norm
        
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        if diff < tol:
            points = new_points
            break
        
        points = new_points
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr


def _simulated_annealing(points, time_limit=None, start_time=None, seed=42):
    """Simulated annealing."""
    points = _normalize(points)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    rng = np.random.RandomState(seed)
    T = 0.03
    cooling = 0.997
    T_min = 1e-7
    
    current_cr = best_cr
    current_points = points.copy()
    
    while T > T_min:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        n_pert = rng.randint(1, 4)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = current_points.copy()
        for idx in indices:
            noise = rng.randn(3) * T
            perturbed[idx] = current_points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        new_cr = _covering_radius(perturbed)
        delta = new_cr - current_cr
        
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            current_points = perturbed
            current_cr = new_cr
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = perturbed.copy()
        
        T *= cooling
    
    return best_points, best_cr


def _multi_beta_lse(points, time_limit=None, start_time=None):
    """Multi-stage LSE refinement with decreasing beta."""
    current = points
    current_cr = _covering_radius(current)
    best = current.copy()
    best_cr = current_cr
    
    betas = [5.0, 10.0, 20.0, 40.0, 80.0, 150.0]
    for beta in betas:
        if time_limit is not None and start_time is not None:
            remaining = time_limit - (time.time() - start_time)
            if remaining < 3:
                break
        pts, cr = _lbfgs_smooth(current, beta, time_limit=time_limit, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()
        current = pts
    
    return best, best_cr


def _refine(points, time_limit=None, start_time=None):
    """Multi-stage: Lloyd -> Riesz polish -> multi-beta LSE."""
    # Stage 1: Lloyd
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, _covering_radius(points)
        lloyd_time = min(remaining * 0.25, 10)
    else:
        lloyd_time = 10
    
    points, cr1 = _lloyd_iteration(points, max_iters=200, time_limit=lloyd_time, start_time=start_time)
    
    # Stage 2: Riesz polish (diversify basin)
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return points, cr1
        riesz_time = min(remaining * 0.25, 12)
    else:
        riesz_time = 12
    
    riesz_pts, riesz_cr = _lbfgs_riesz(points, s=2.0, time_limit=riesz_time, start_time=start_time)
    # Combine: take the better of riesz result and lloyd result
    if riesz_cr < cr1:
        points = riesz_pts
        cr1 = riesz_cr
    
    # Stage 3: Multi-beta LSE
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return points, cr1
        lse_time = remaining * 0.5
    else:
        lse_time = 25
    
    points, cr2 = _multi_beta_lse(points, time_limit=lse_time, start_time=start_time)
    
    return points, cr2


def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _jittered_fibonacci_points(n, seed=42, jitter=0.02):
    """Fibonacci points with small random jitter."""
    pts = _fibonacci_points(n)
    rng = np.random.RandomState(seed)
    noise = rng.randn(n, 3) * jitter
    pts = pts + noise
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _random_points(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _icosahedron_based_points(n):
    """Generate points based on icosahedron symmetry with perturbations."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    
    pts = base.copy()
    rng = np.random.RandomState(123)
    while len(pts) < n:
        i, j = rng.randint(0, len(pts), 2)
        if i == j:
            continue
        mid = (pts[i] + pts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid /= mid_norm
            dists = 1.0 - np.sum(mid * pts, axis=1)
            if np.min(dists) > 0.01:
                pts = np.vstack([pts, mid])
    
    return pts[:n]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate multiple diverse initial configurations
    initial_configs = [
        _fibonacci_points(N),
        _jittered_fibonacci_points(N, seed=1, jitter=0.03),
        _jittered_fibonacci_points(N, seed=2, jitter=0.05),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _icosahedron_based_points(N),
    ]
    
    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)
    
    # Phase 1: Refine each initial configuration with time budget
    n_configs = len(initial_configs)
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        time_per_config = min(remaining / (n_configs - i), 25)
        refined, cr = _refine(init_pts, time_limit=time_per_config, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()
    
    # Phase 2: Simulated annealing from best configuration
    remaining = deadline - time.time()
    if remaining > 15:
        sa_time = min(remaining * 0.3, 25)
        sa_points, sa_cr = _simulated_annealing(best_points, time_limit=sa_time, start_time=start_time, seed=123)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_points.copy()
    
    # Phase 3: Final multi-beta LSE refinement
    remaining = deadline - time.time()
    if remaining > 10:
        final_time = min(remaining * 0.7, 30)
        final_points, final_cr = _multi_beta_lse(best_points, time_limit=final_time, start_time=start_time)
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = final_points.copy()
    
    # Phase 4: One more Lloyd pass for final polish
    remaining = deadline - time.time()
    if remaining > 5:
        lloyd_points, lloyd_cr = _lloyd_iteration(best_points, max_iters=100, time_limit=remaining * 0.5, start_time=start_time)
        if lloyd_cr < best_cr:
            best_cr = lloyd_cr
            best_points = lloyd_points.copy()
    
    # Normalize final result
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
