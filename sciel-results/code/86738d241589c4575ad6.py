"""Sphere covering with 50 points - optimized algorithm."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.special import lpmv

def _covering_radius(points):
    """Compute covering radius in radians via convex-hull facet circumcenters."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    try:
        hull = ConvexHull(points)
    except Exception:
        from scipy.spatial import cKDTree
        rng = np.random.RandomState(0)
        tree = cKDTree(points)
        n_samples = 50000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return float(np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1))))

    max_r = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    return float(max_r)

def _spherical_harmonic_basis(max_l=10):
    """Precompute spherical harmonic basis functions up to degree max_l."""
    # We'll store Y_l^m for m=0, |m|=1, ..., l
    # Using real spherical harmonics for simplicity
    basis = []  # list of (l, m, coeff, func)
    
    for l in range(max_l + 1):
        for m in range(0, l + 1):
            # Compute normalization constant
            if m == 0:
                norm = np.sqrt((2*l+1)/(4*np.pi) * np.prod([(2*k-1)/(2*k) for k in range(1, l+1)]))
            else:
                norm = np.sqrt(2) * np.sqrt((2*l+1)/(4*np.pi) * np.prod([(2*k-1)/(2*k) for k in range(1, l+1)]))
            
            # Associated Legendre function P_l^m(cos(theta))
            def P_lm(l, m, cos_theta):
                if m > l:
                    return 0.0
                if m == 0:
                    return lpmv(0, l, cos_theta)
                else:
                    return lpmv(m, l, cos_theta)
            
            # Store as a lambda that takes (theta, phi)
            def make_func(l, m, norm):
                def func(theta, phi):
                    P = P_lm(l, m, np.cos(theta))
                    if m == 0:
                        return norm * P
                    else:
                        return norm * P * np.cos(m * phi)
                return func
            
            basis.append((l, m, norm, make_func(l, m, norm)))
    
    return basis

def _fast_facets(points):
    """Return list of (radius, circumcenter, facet_indices) for triangular facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    out = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        out.append((r, v, simplex))
    return out

def _riesz_energy(points, s=12.0):
    """Compute Riesz s-energy: sum over all pairs of 1/d^s where d is angular distance."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    # Compute all pairwise dot products
    dots = points @ points.T
    # Clip for numerical stability
    dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    # Angular distance
    angles = np.arccos(dots)
    # Energy: sum over i<j of 1/angle^s
    iu = np.triu_indices(n, k=1)
    ang_ij = angles[iu]
    # Avoid division by zero
    ang_ij = np.maximum(ang_ij, 1e-15)
    energy = np.sum(ang_ij ** (-s))
    return energy

def _distance_to_nearest(points, max_l=8, n_theta=50, n_phi=100):
    """
    Compute distance to nearest point using spherical harmonic expansion.
    Returns:
        - dist_field: array of shape (n_theta, n_phi) with angular distances
        - theta: array of shape (n_theta,) 
        - phi: array of shape (n_phi,)
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    
    # Create evaluation grid
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)
    eval_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Compute angular distances to each source point
    # dist_ij = arccos(dot(eval_pt, source_j))
    dots = eval_pts @ points.T  # (n_eval, n)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (n_eval, n)
    
    # Distance to nearest
    dist_field = np.min(angles, axis=1).reshape(n_theta, n_phi)
    
    return dist_field, theta, phi

def _riesz_energy_grad(points, s=12.0):
    """
    Compute Riesz s-energy and its gradient w.r.t. points.
    E = sum_{i<j} 1/d_ij^s where d_ij = arccos(p_i . p_j)
    dE/dp_i = -s * sum_{j!=i} (1/d_ij^s) * d(1/d_ij^s)/d(d_ij) * dd_ij/dp_i
    d(1/d^s)/d(d) = -s * d^(-s-1)
    d(d_ij)/d(p_i) where d_ij = arccos(p_i.p_j):
      = -1/sin(d_ij) * d(p_i.p_j)/d(p_i) = -1/sin(d_ij) * p_j
    So dE/dp_i = -s * sum_j (1/d_ij^s) * (-s * d_ij^(-s-1)) * (-p_j/sin(d_ij))
               = -s^2 * sum_j d_ij^(-2s-1) * p_j/sin(d_ij)
    Wait, let me recompute more carefully.
    
    E = sum_{i<j} f(d_ij) where f(d) = d^(-s)
    f'(d) = -s * d^(-s-1)
    
    d_ij = arccos(p_i . p_j)
    d(d_ij)/d(p_i) = -1/sqrt(1-(p_i.p_j)^2) * p_j = -p_j/sin(d_ij)
    
    dE/dp_i = sum_{j!=i} f'(d_ij) * d(d_ij)/d(p_i)
            = sum_{j!=i} (-s * d_ij^(-s-1)) * (-p_j/sin(d_ij))
            = s * sum_{j!=i} d_ij^(-s-1) * p_j/sin(d_ij)
    
    But we want to move points to MINIMIZE energy (spread them out),
    so gradient descent: p_i -= lr * dE/dp_i
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    dots = points @ points.T
    dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    angles = np.arccos(dots)
    sins = np.sin(angles)
    sins = np.maximum(sins, 1e-15)
    
    # f'(d_ij) * d(d_ij)/d(p_i) for each pair
    # Contribution to grad[i] from point j: s * d_ij^(-s-1) / sin(d_ij) * p_j
    # d_ij^(-s-1) = angles^(-s-1)
    ang_powers = angles ** (-s - 1)
    # Set diagonal to 0
    np.fill_diagonal(ang_powers, 0.0)
    np.fill_diagonal(sins, 1.0)  # avoid div by zero on diagonal
    np.fill_diagonal(dots, 1.0)
    
    # weight_ij = s * angles^(-s-1) / sin(angles)
    weights = s * ang_powers / sins
    np.fill_diagonal(weights, 0.0)
    
    # grad[i] = sum_j weights[i,j] * p_j
    grad = weights @ points
    
    # Energy value
    iu = np.triu_indices(n, k=1)
    energy = np.sum(angles[iu] ** (-s))
    
    # Project gradient to tangent space
    grad_tan = grad - np.sum(grad * points, axis=1, keepdims=True) * points
    
    return energy, grad_tan

def _lbfgs_riesz(points, max_time=20.0, start_time=None, s=12.0, maxiter=300):
    """
    Minimize Riesz s-energy via L-BFGS-B on the sphere (via normalization).
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x_norms = np.linalg.norm(x, axis=1, keepdims=True)
        x_norms[x_norms < 1e-15] = 1.0
        x = x / x_norms
        val, grad = _riesz_energy_grad(x, s=s)
        # Project gradient to tangent space
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
        )
        x_opt = result.x.reshape(-1, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        cr_opt = _covering_radius(x_opt)

        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    return best, best_cr

def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Compute the smooth covering radius and its gradient w.r.t. points.
    Uses log-sum-exp over facet circumradii.
    Returns (value, gradient) where gradient has shape (N, 3).
    """
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    try:
        hull = ConvexHull(points)
    except Exception:
        from scipy.spatial import cKDTree
        rng = np.random.RandomState(0)
        tree = cKDTree(points)
        n_samples = 30000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        ang_dists = 2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1))
        m = np.max(ang_dists)
        if m < 1e-12:
            return 0.0, np.zeros_like(points)
        val = m + np.log(np.sum(np.exp(beta * (ang_dists - m)))) / beta
        return val, np.zeros_like(points)

    radii = []
    centers = []
    simplices = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        radii.append(r)
        centers.append(v)
        simplices.append(simplex)

    if not radii:
        return 0.0, np.zeros_like(points)
    radii = np.array(radii)
    m = np.max(radii)
    if m < 1e-12:
        return 0.0, np.zeros_like(points)
    val = m + np.log(np.sum(np.exp(beta * (radii - m)))) / beta

    w = np.exp(beta * (radii - m))
    w /= w.sum()

    grad = np.zeros_like(points)
    for i in range(len(radii)):
        center = centers[i]
        simplex = simplices[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            g = -center / sin_r
            g_tan = g - np.dot(g, p) * p
            grad[idx] += w[i] * g_tan

    return val, grad

def _sh_covering_radius_grad(points, beta=50.0, n_theta=40, n_phi=80):
    """
    Compute smooth covering radius using spherical harmonic representation.
    This avoids convex hull recomputation and provides a smoother landscape.
    
    Approach:
    1. Compute distance-to-nearest field on a grid
    2. Use log-sum-exp to get smooth max
    3. Compute gradient via backprop through the min operation
    
    Returns (value, gradient) where gradient has shape (N, 3).
    """
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    n = len(points)
    
    # Create evaluation grid
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)
    eval_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Compute angular distances: dist[i,j] = arccos(eval_pts[i] . points[j])
    dots = eval_pts @ points.T  # (n_eval, n)
    dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    angles = np.arccos(dots)  # (n_eval, n)
    
    # For each evaluation point, find which source is nearest
    min_angles = np.min(angles, axis=1)
    
    # Smooth max via log-sum-exp
    m = np.max(min_angles)
    if m < 1e-12:
        return 0.0, np.zeros_like(points)
    
    # Weight for each eval point
    w_eval = np.exp(beta * (min_angles - m))
    w_eval /= w_eval.sum()
    
    val = m + np.log(np.sum(np.exp(beta * (min_angles - m)))) / beta
    
    # Gradient: for each source point j, sum over eval points i where j is nearest
    # d(min_angle_i)/d(points[j]) = -points[j] / sin(angle_ij) if j is the argmin
    # Using soft assignment: weight_i_j = softmax over j for each i
    
    # Compute soft assignment weights
    # For numerical stability, subtract max per row
    neg_angles = -angles  # (n_eval, n)
    max_neg = np.max(neg_angles, axis=1, keepdims=True)
    exp_neg = np.exp(beta * (neg_angles - max_neg))
    sum_exp = np.sum(exp_neg, axis=1, keepdims=True)
    assign = exp_neg / sum_exp  # (n_eval, n) - probability that source j is nearest for eval i
    
    # Gradient contribution from each (eval, source) pair
    # d(min_angle_i)/d(points[j]) ≈ assign[i,j] * (-points[j] / sin(angle_ij))
    # Project to tangent space
    
    grad = np.zeros_like(points)
    
    # Compute sin(angles)
    sin_angles = np.sin(angles)  # (n_eval, n)
    sin_angles = np.maximum(sin_angles, 1e-15)
    
    # Weight matrix: w_eval[i] * assign[i,j] / sin(angle_ij)
    # Shape: (n_eval, n)
    weight_mat = (w_eval[:, None] * assign) / sin_angles
    
    # grad[j] = sum_i weight_mat[i,j] * (-points[j])
    # But we need to project to tangent space
    # grad[j] = -points[j] * sum_i weight_mat[i,j]
    # Then project: grad_tan[j] = grad[j] - (grad[j].points[j]) * points[j]
    
    # sum_i weight_mat[i,j] for each j
    total_weights = np.sum(weight_mat, axis=0)  # (n,)
    
    # Raw gradient
    raw_grad = -total_weights[:, None] * points  # (n, 3)
    
    # Project to tangent space
    grad_proj = raw_grad - np.sum(raw_grad * points, axis=1, keepdims=True) * points
    
    return val, grad_proj

def _lbfgs_refine(points, max_time=20.0, start_time=None, beta=50.0, use_sh=False):
    """
    Refine points using L-BFGS-B on a smooth log-sum-exp surrogate for the covering radius.
    Uses facet-based LSE gradient (accurate and fast for N=50).
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        val, grad = _smooth_covering_radius_grad(x, beta=beta)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 600}
        )
        x_opt = result.x.reshape(-1, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        cr_opt = _covering_radius(x_opt)

        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    return best, best_cr

def _spherical_gd(points, max_iters=2000, lr_init=0.008, lr_min=1e-9,
                  time_limit=None, start_time=None):
    """Spherical gradient descent with adaptive step size using covering radius gradient."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    best = points.copy()
    best_cr = _covering_radius(points)
    lr = lr_init
    no_improve = 0

    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        grad, cur_cr = _grad_covering_radius(points)
        gn = np.linalg.norm(grad)
        if gn < 1e-14:
            break
        grad /= gn
        accepted = False
        for trial in range(5):
            cand = points + lr * grad
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            cand_cr = _covering_radius(cand)
            if cand_cr < best_cr - 1e-14:
                points = cand
                best = cand.copy()
                best_cr = cand_cr
                lr = min(lr * 1.05, 0.05)
                no_improve = 0
                accepted = True
                break
            lr = lr * 0.7
        if not accepted:
            no_improve += 1
            lr = max(lr * 0.85, lr_min)
            if no_improve > 80 or lr < lr_min * 1.01:
                break
    return best, best_cr

def _grad_covering_radius(points):
    """
    Compute a gradient of the covering radius w.r.t. the points.
    Uses the top-K largest facets with soft weighting.
    """
    facets = _fast_facets(points)
    if not facets:
        return np.zeros_like(points), 0.0
    facets.sort(key=lambda x: -x[0])
    max_r = facets[0][0]
    K = min(10, len(facets))
    radii = np.array([f[0] for f in facets[:K]])
    beta = 30.0
    w = np.exp(beta * (radii - max_r))
    w /= w.sum()
    grad = np.zeros_like(points)
    for i in range(K):
        r, center, simplex = facets[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            g = -center / sin_r
            g_tan = g - np.dot(g, p) * p
            grad[idx] += w[i] * g_tan
    return grad, max_r

def _spherical_lloyd(points, max_iters=200, time_limit=None, start_time=None):
    """
    Lloyd-style: each point moves to the average circumcenter of its adjacent facets.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    best = points.copy()
    best_cr = _covering_radius(points)

    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        facets = _fast_facets(points)
        if not facets:
            break
        centers_per_point = [[] for _ in range(len(points))]
        for r, center, simplex in facets:
            for idx in simplex:
                centers_per_point[idx].append(center)
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            if not centers_per_point[i]:
                new_points[i] = points[i]
            else:
                c = np.mean(centers_per_point[i], axis=0)
                cn = np.linalg.norm(c)
                if cn < 1e-12:
                    new_points[i] = points[i]
                else:
                    new_points[i] = c / cn
        alpha = 0.5
        blended = (1 - alpha) * points + alpha * new_points
        blended /= np.linalg.norm(blended, axis=1, keepdims=True)
        cr = _covering_radius(blended)
        if cr < best_cr:
            points = blended
            best = points.copy()
            best_cr = cr
        else:
            alpha = 0.2
            blended = (1 - alpha) * points + alpha * new_points
            blended /= np.linalg.norm(blended, axis=1, keepdims=True)
            cr = _covering_radius(blended)
            if cr < best_cr:
                points = blended
                best = points.copy()
                best_cr = cr
            else:
                break
    return best, best_cr

def _simulated_annealing(points, time_limit=None, start_time=None, seed=42,
                         T_init=0.06, T_min=1e-6, cooling=0.997):
    """Simulated annealing with targeted perturbations on worst facet vertices."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    best = points.copy()
    best_cr = _covering_radius(points)
    rng = np.random.RandomState(seed)
    T = T_init
    cur = points.copy()
    cur_cr = best_cr

    while T > T_min:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        # Get worst facet to target perturbations
        facets = _fast_facets(cur)
        if facets:
            facets.sort(key=lambda x: -x[0])
            worst = facets[0]
            target_idx = set(worst[2].tolist())
            # Add neighbors from second-worst facet
            if len(facets) > 1:
                target_idx.update(facets[1][2].tolist())
        else:
            target_idx = set(range(N))
        
        n_pert = rng.randint(1, 4)
        # Bias selection toward worst facet vertices
        target_list = list(target_idx)
        if rng.rand() < 0.6 and target_list:
            idxs = list(set(rng.choice(target_list, size=min(n_pert, len(target_list)), replace=False)))
            if len(idxs) < n_pert:
                remaining = [i for i in range(N) if i not in idxs]
                extra = rng.choice(remaining, size=n_pert - len(idxs), replace=False)
                idxs = idxs + list(extra)
        else:
            idxs = rng.choice(N, size=n_pert, replace=False)
        
        cand = cur.copy()
        for i in idxs:
            noise = rng.randn(3) * T
            cand[i] = cur[i] + noise
            cand[i] /= np.linalg.norm(cand[i])
        cand_cr = _covering_radius(cand)
        delta = cand_cr - cur_cr
        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-12)):
            cur = cand
            cur_cr = cand_cr
            if cand_cr < best_cr:
                best = cand.copy()
                best_cr = cand_cr
        T *= cooling
    return best, best_cr

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _icosahedron_based_points(n):
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
        mn = np.linalg.norm(mid)
        if mn > 1e-10:
            mid /= mn
            dists = 1.0 - np.sum(mid * pts, axis=1)
            if np.min(dists) > 0.01:
                pts = np.vstack([pts, mid])
    return pts[:n]

def _repulsion_refine(points, max_iters=500, time_limit=None, start_time=None):
    """
    Refine using repulsive forces: each point is pushed away from others.
    Uses a combination of Riesz energy gradient and direct repulsion.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    best = points.copy()
    best_cr = _covering_radius(points)
    lr = 0.01

    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Compute repulsive gradient using Riesz energy
        _, grad = _riesz_energy_grad(points, s=8.0)
        gn = np.linalg.norm(grad)
        if gn < 1e-14:
            break
        grad /= gn
        
        # Try step
        cand = points + lr * grad
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        cand_cr = _covering_radius(cand)
        
        if cand_cr < best_cr - 1e-14:
            points = cand
            best = cand.copy()
            best_cr = cand_cr
            lr = min(lr * 1.02, 0.05)
        else:
            lr *= 0.9
            if lr < 1e-8:
                break
    
    return best, best_cr

def _refine(points, time_limit=None, start_time=None):
    """Multi-stage refinement: Riesz L-BFGS -> Lloyd -> L-BFGS covering -> GD."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    # Stage 1: Riesz energy L-BFGS (good for spreading)
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 3:
            return points, _covering_radius(points)
        t1 = min(rem * 0.3, 25.0)
    else:
        t1 = 25.0
    points, cr = _lbfgs_riesz(points, max_time=t1, start_time=start_time, s=12.0)

    # Stage 2: Lloyd's algorithm
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, cr
        t2 = min(rem * 0.2, 8.0)
    else:
        t2 = 8.0
    points, cr = _spherical_lloyd(points, max_iters=100, time_limit=t2, start_time=start_time)

    # Stage 3: L-BFGS with smooth covering radius surrogate (facet-based)
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 3:
            return points, cr
        t3 = min(rem * 0.3, 25.0)
    else:
        t3 = 25.0
    points, cr = _lbfgs_refine(points, max_time=t3, start_time=start_time, beta=50.0)

    # Stage 4: Lloyd again
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, cr
        t4 = min(rem * 0.15, 6.0)
    else:
        t4 = 6.0
    points, cr = _spherical_lloyd(points, max_iters=80, time_limit=t4, start_time=start_time)

    # Stage 5: L-BFGS polish with higher beta (facet-based)
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 3:
            return points, cr
        t5 = min(rem * 0.25, 20.0)
    else:
        t5 = 20.0
    points, cr = _lbfgs_refine(points, max_time=t5, start_time=start_time, beta=100.0)

    # Stage 6: Final GD polish
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 1:
            return points, cr
        t6 = max(rem * 0.5, 1.0)
    else:
        t6 = 5.0
    points, cr = _spherical_gd(points, max_iters=1000, lr_init=0.004,
                               time_limit=t6, start_time=start_time)
    return points, cr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    initial_configs = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=7),
        _random_points(N, seed=13),
        _icosahedron_based_points(N),
        _random_points(N, seed=99),
        _random_points(N, seed=200),
    ]

    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)

    # Phase 1: Refine each initial config with a time budget
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        time_per = min(remaining / (len(initial_configs) - i), 20.0)
        refined, cr = _refine(init_pts, time_limit=time_per, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()

    # Phase 2: Simulated annealing from best
    remaining = deadline - time.time()
    if remaining > 15:
        sa_time = min(remaining * 0.25, 20.0)
        sa_pts, sa_cr = _simulated_annealing(best_points, time_limit=sa_time,
                                             start_time=start_time, seed=777)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_pts.copy()

    # Phase 3: Final refinement
    remaining = deadline - time.time()
    if remaining > 10:
        fin_time = min(remaining * 0.4, 30.0)
        fin_pts, fin_cr = _refine(best_points, time_limit=fin_time, start_time=start_time)
        if fin_cr < best_cr:
            best_cr = fin_cr
            best_points = fin_pts.copy()

    # Phase 4: Riesz energy polish
    remaining = deadline - time.time()
    if remaining > 8:
        riesz_time = max(remaining * 0.3, 5.0)
        riesz_pts, riesz_cr = _lbfgs_riesz(best_points, max_time=riesz_time,
                                           start_time=start_time, s=16.0, maxiter=400)
        if riesz_cr < best_cr:
            best_cr = riesz_cr
            best_points = riesz_pts.copy()

    # Phase 5: Final L-BFGS polish with high beta (facet-based for accuracy)
    remaining = deadline - time.time()
    if remaining > 5:
        lbfgs_time = max(remaining * 0.5, 3.0)
        lbfgs_pts, lbfgs_cr = _lbfgs_refine(best_points, max_time=lbfgs_time,
                                            start_time=start_time, beta=200.0)
        if lbfgs_cr < best_cr:
            best_cr = lbfgs_cr
            best_points = lbfgs_pts.copy()

    # Phase 5b: Very high beta polish for final precision
    remaining = deadline - time.time()
    if remaining > 4:
        lbfgs_time2 = max(remaining * 0.5, 2.0)
        lbfgs_pts2, lbfgs_cr2 = _lbfgs_refine(best_points, max_time=lbfgs_time2,
                                              start_time=start_time, beta=500.0)
        if lbfgs_cr2 < best_cr:
            best_cr = lbfgs_cr2
            best_points = lbfgs_pts2.copy()

    # Final GD micro-polish
    remaining = deadline - time.time()
    if remaining > 2:
        gd_time = max(remaining * 0.8, 1.0)
        gd_pts, gd_cr = _spherical_gd(best_points, max_iters=500, lr_init=0.002,
                                      time_limit=gd_time, start_time=start_time)
        if gd_cr < best_cr:
            best_cr = gd_cr
            best_points = gd_pts.copy()

    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
# EVOLVE-BLOCK-END
