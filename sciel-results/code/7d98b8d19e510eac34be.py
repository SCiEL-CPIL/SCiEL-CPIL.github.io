"""Packing 200 equal circles in a unit square: L-BFGS-B optimization with pdist."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 200
DEADLINE = 145.0  # seconds; leave margin before 150s kill


def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Distance to boundary
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances using pdist (faster and more memory efficient)
    dists = pdist(pts)
    min_dist = np.min(dists)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0


def _soft_min_neg(x, alpha):
    """
    Objective: -soft_min of all relevant distances.
    x: flat array of size 2N.
    Returns scalar objective value.
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()
    
    # Pairwise distances using pdist
    pairwise_dists = pdist(pts)
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Soft min: min(d_i) ≈ -1/alpha * log(sum(exp(-alpha * d_i)))
    m = np.min(all_dists)
    log_sum = m + np.log(np.sum(np.exp(-alpha * (all_dists - m))))
    soft_min_val = -(1.0 / alpha) * log_sum
    
    # We want to maximize min distance, so minimize -min distance
    return -soft_min_val


def _soft_min_neg_grad(x, alpha):
    """
    Analytical gradient of _soft_min_neg.
    x: flat array of size 2N.
    Returns (objective, gradient).
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()
    
    # Pairwise distances using pdist
    pairwise_dists = pdist(pts)
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Soft min computation
    m = np.min(all_dists)
    shifted = -alpha * (all_dists - m)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = m + np.log(sum_exp)
    soft_min_val = -(1.0 / alpha) * log_sum
    
    # Weights for each distance term in the soft min
    weights = exp_shifted / sum_exp  # sum to 1
    
    # Gradient of soft_min with respect to each distance
    # d(soft_min)/d(d_i) = -weights[i]
    # So d(-soft_min)/d(d_i) = weights[i]
    
    # Now compute gradient w.r.t. x
    grad = np.zeros(2 * N)
    
    # Boundary distance gradients
    # boundary_dists[i] = min(0.5 - |x_i|, 0.5 - |y_i|)
    # d(0.5 - |x_i|)/dx_i = -sign(x_i)
    # d(0.5 - |y_i|)/dy_i = -sign(y_i)
    
    # For each point, determine which boundary constraint is active
    for i in range(N):
        w_b = weights[i]  # weight for this point's boundary distance
        
        # Which of d_x[i] or d_y[i] is the minimum?
        if d_x[i] <= d_y[i]:
            # boundary = 0.5 - |x_i|
            if pts[i, 0] > 0:
                grad[2*i] += w_b * (-1.0)  # d(-soft_min)/dx_i = w_b * (-sign(x_i))
            elif pts[i, 0] < 0:
                grad[2*i] += w_b * (1.0)
            # else pts[i,0] == 0, subgradient, use 0
        else:
            # boundary = 0.5 - |y_i|
            if pts[i, 1] > 0:
                grad[2*i+1] += w_b * (-1.0)
            elif pts[i, 1] < 0:
                grad[2*i+1] += w_b * (1.0)
    
    # Pairwise distance gradients
    # pairwise_dists[k] corresponds to pair (i, j) where i < j
    # Need to map pdist index to (i, j)
    # pdist uses condensed form: index k corresponds to (i, j) where
    # i = floor((k - N + 1 + sqrt(8*N^2 - 8*N + 1 - 8*k)) / 2) ... complicated
    # Instead, build the full distance matrix and use its gradient
    
    # Build full pairwise distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    
    # The pairwise_dists from pdist are the upper triangle of dist_mat
    # weights[N + k] corresponds to dist_mat[i, j] for the k-th pair in condensed order
    
    # Easier: compute gradient contribution from full matrix
    # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij
    # d(dist_ij)/dx_j = (x_j - x_i) / dist_ij
    
    # For each pair (i, j), i < j:
    # weight = weights[N + pair_index]
    # grad[2*i] += weight * (pts[i] - pts[j]) / dist_ij
    # grad[2*j] += weight * (pts[j] - pts[i]) / dist_ij
    
    # Use the condensed form mapping
    # pdist condensed index k: k = i*(N-1) - i*(i+1)//2 + j - i - 1 for i < j
    # Let's just iterate
    pair_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = dist_mat[i, j]
            if d > 1e-12:
                w = weights[N + pair_idx]
                diff_ij = pts[i] - pts[j]
                grad[2*i] += w * diff_ij[0] / d
                grad[2*i+1] += w * diff_ij[1] / d
                grad[2*j] += w * diff_ij[0] / d
                grad[2*j+1] += w * diff_ij[1] / d
            pair_idx += 1
    
    obj = -soft_min_val
    return obj, grad


def _hex_init():
    """Generate a hexagonal lattice initialization."""
    best_pts = None
    best_r = 0
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.9, s_est * 1.1, 10):
        pts_list = []
        row = 0
        while True:
            y = -0.5 + s / 2 + row * s * np.sqrt(3) / 2
            if y > 0.5 - s / 2:
                break
            offset = (row % 2) * s / 2
            x = -0.5 + s / 2 + offset
            while x <= 0.5 - s / 2:
                pts_list.append((x, y))
                x += s
            row += 1
        if len(pts_list) < N:
            continue
        pts = np.array(pts_list[:N])
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts


def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    return pts[:N].copy()


def _random_init(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _objective(x, alpha):
    """
    Objective function for L-BFGS-B.
    x: flat array of size 2N.
    alpha: sharpness of the soft-min.
    Returns: -soft_min(distances) to minimize.
    """
    pts = x.reshape(N, 2)

    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()

    # Pairwise distances using pdist
    pairwise_dists = pdist(pts)

    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])

    # Soft min: min(d_i) ≈ -1/alpha * log(sum(exp(-alpha * d_i)))
    # Use log-sum-exp trick for numerical stability
    m = np.min(all_dists)
    log_sum = m + np.log(np.sum(np.exp(-alpha * (all_dists - m))))
    soft_min_val = -(1.0 / alpha) * log_sum

    # We want to maximize min distance, so minimize -min distance
    return -soft_min_val


def _optimize(pts, time_budget, alphas=[20.0, 50.0, 100.0]):
    """
    Run L-BFGS-B optimization on the initial points with increasing alpha.
    Uses analytical gradients for speed.
    """
    t0 = time.time()
    x0 = pts.flatten()

    for alpha in alphas:
        if time.time() - t0 > time_budget:
            break

        def obj(x):
            val, _ = _soft_min_neg_grad(x, alpha)
            return val

        def obj_grad(x):
            _, g = _soft_min_neg_grad(x, alpha)
            return g

        try:
            res = minimize(
                obj, x0, jac=obj_grad, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'gtol': 1e-6, 'maxls': 30}
            )
            x0 = res.x
        except Exception:
            pass

    return x0.reshape(N, 2)


def _active_set_polish(pts, time_budget, tol=1e-10):
    """
    Polish using only the active (tight) constraints.
    Identify pairs and boundary constraints that are near the minimum,
    then optimize only those.
    """
    t0 = time.time()
    pts = pts.copy()
    
    for iteration in range(50):
        if time.time() - t0 > time_budget:
            break
        
        r = _score(pts)
        if r <= 0:
            break
        
        # Find active constraints
        ax = np.abs(pts[:, 0])
        ay = np.abs(pts[:, 1])
        boundary = np.minimum(0.5 - ax, 0.5 - ay)
        
        # Active boundary constraints: within 2*r of boundary
        active_boundary = np.where(boundary < 2.0 * r + 1e-9)[0]
        
        # Active pairwise constraints
        dists = pdist(pts)
        active_pairs = np.where(dists < 2.0 * r + 1e-9)[0]
        
        # If no active constraints, we're at a local optimum
        if len(active_boundary) == 0 and len(active_pairs) == 0:
            break
        
        # Build list of active constraints
        # We'll use a simple gradient-based approach on active constraints
        
        # For simplicity, use L-BFGS-B with a tighter objective
        x0 = pts.flatten()
        
        def obj_active(x):
            p = x.reshape(N, 2)
            
            # Compute all distances
            d_x = 0.5 - np.abs(p[:, 0])
            d_y = 0.5 - np.abs(p[:, 1])
            b_dists = np.minimum(d_x, d_y).flatten()
            p_dists = pdist(p)
            all_d = np.concatenate([b_dists, p_dists])
            
            # Penalize any distance below current r
            violations = np.maximum(0, r - all_d)
            return np.sum(violations**2) + 1e-4 * np.sum(np.maximum(0, all_d - r)**2)
        
        try:
            res = minimize(
                obj_active, x0, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 50, 'gtol': tol}
            )
            new_pts = res.x.reshape(N, 2)
            new_pts = np.clip(new_pts, -0.5, 0.5)
            new_r = _score(new_pts)
            if new_r >= r - 1e-12:
                pts = new_pts
            else:
                break
        except Exception:
            break
    
    return pts


def construct():
    t_start = time.time()

    initializers = [
        ("hex", _hex_init),
        ("grid", _grid_init),
        ("rand0", lambda: _random_init(0)),
        ("rand1", lambda: _random_init(1)),
        ("rand2", lambda: _random_init(42)),
    ]

    best_pts = None
    best_r = 0.0

    for name, init_fn in initializers:
        if time.time() > DEADLINE - 15:
            break

        pts = init_fn()
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

        # Optimize
        remaining = DEADLINE - time.time()
        if remaining < 8:
            break

        # Allocate time based on initializer quality
        if name == "hex":
            time_budget = min(remaining * 0.35, 35.0)
        elif name == "grid":
            time_budget = min(remaining * 0.20, 20.0)
        else:
            time_budget = min(remaining * 0.15, 15.0)

        opt_pts = _optimize(pts, time_budget)
        opt_pts = np.clip(opt_pts, -0.5, 0.5)

        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()

    # Final polish: high-alpha optimization + active set polish
    remaining = DEADLINE - time.time()
    if remaining > 10 and best_pts is not None:
        # Phase 1: high-alpha L-BFGS-B
        time_budget1 = min(remaining * 0.4, 12.0)
        opt_pts = _optimize(best_pts, time_budget1, alphas=[100.0, 200.0])
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()

    # Phase 2: Active set polish
    remaining = DEADLINE - time.time()
    if remaining > 5 and best_pts is not None:
        time_budget2 = min(remaining * 0.5, 10.0)
        polished = _active_set_polish(best_pts, time_budget2)
        polished = np.clip(polished, -0.5, 0.5)
        pol_r = _score(polished)
        if pol_r > best_r:
            best_r = pol_r
            best_pts = polished.copy()

    if best_pts is None:
        best_pts = _hex_init()

    return best_pts
