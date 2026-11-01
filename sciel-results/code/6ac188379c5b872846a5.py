"""Optimize 50 circle centres in the unit disk via soft-min + L-BFGS-B with high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 130.0  # Safety margin before 150s limit


def _compute_score(points):
    """Compute the packing radius (higher is better)."""
    if points.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Boundary: each centre inside unit circle
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_margin = 1.0 - np.max(dists)
    if boundary_margin <= 0:
        return 0.0
    # Pairwise distances
    d = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            diff = points[i] - points[j]
            d[i, j] = d[j, i] = np.sqrt(np.dot(diff, diff))
    min_pair = np.min(d[np.triu_indices(N, k=1)])
    if min_pair <= 0:
        return 0.0
    return min(min_pair / 2.0, boundary_margin)


def _soft_min(values, beta):
    """Compute soft minimum of values using log-sum-exp trick."""
    # values shape: (M,)
    v_min = np.min(values)
    # For numerical stability
    shifted = values - v_min
    # soft min ≈ v_min - (1/beta) * log(sum(exp(-beta * shifted)))
    # = v_min - (1/beta) * log(sum(exp(beta*(v_min - values))))
    # Let's use: soft_min = v_min - (1/beta)*log(sum(exp(-beta*(values - v_min))))
    # = v_min - (1/beta)*log(sum(exp(-beta*shifted)))
    # Since shifted >= 0, exp(-beta*shifted) <= 1
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    return v_min - log_sum / beta


def _objective_and_gradient(x, beta):
    """
    Objective: negative of soft-min of all constraints (pairwise dists and boundary margins).
    We want to maximize the minimum, so minimize -soft_min.
    
    x: flat array of shape (2N,)
    Returns: (f, grad)
    """
    pts = x.reshape(N, 2)
    
    # Compute all pairwise distances
    # For efficiency, compute all pairwise squared distances
    # d_ij = ||p_i - p_j||
    # Boundary margin: 1 - ||p_i||
    
    # Collect all constraint values
    # Pairwise: for i < j, d_ij
    # Boundary: for each i, 1 - ||p_i||
    
    n_pairs = N * (N - 1) // 2
    all_vals = np.empty(n_pairs + N)
    idx = 0
    
    # Pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            diff = pts[i] - pts[j]
            d = np.sqrt(np.dot(diff, diff))
            all_vals[idx] = d
            idx += 1
    
    # Boundary margins
    for i in range(N):
        r = np.sqrt(np.dot(pts[i], pts[i]))
        all_vals[idx] = 1.0 - r
        idx += 1
    
    # Soft min
    f_val = -_soft_min(all_vals, beta)
    
    # Gradient
    # d(soft_min)/d(all_vals[k]) = exp(-beta*(all_vals[k] - v_min)) / sum(exp(-beta*(all_vals - v_min)))
    v_min = np.min(all_vals)
    shifted = all_vals - v_min
    exp_vals = np.exp(-beta * shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros(2 * N)
    idx = 0
    
    # Pairwise gradient contribution
    for i in range(N):
        for j in range(i + 1, N):
            diff = pts[i] - pts[j]
            d = np.sqrt(np.dot(diff, diff))
            if d > 1e-12:
                # d(d_ij)/d(pts[i]) = diff / d_ij
                # d(d_ij)/d(pts[j]) = -diff / d_ij
                w = weights[idx]
                grad[2 * i:2 * i + 2] -= w * diff / d  # because we're minimizing -soft_min
                grad[2 * j:2 * j + 2] += w * diff / d
            idx += 1
    
    # Boundary gradient contribution
    for i in range(N):
        r = np.sqrt(np.dot(pts[i], pts[i]))
        if r > 1e-12:
            # d(1 - r)/d(pts[i]) = -pts[i] / r
            w = weights[idx]
            grad[2 * i:2 * i + 2] += w * pts[i] / r  # minimize -soft_min, so -d(soft_min)/dx = -w * d(constraint)/dx
            # Wait, let me be careful.
            # f = -soft_min(all_vals)
            # df/dx = -sum_k (d(soft_min)/d(all_vals[k])) * d(all_vals[k])/dx
            # d(soft_min)/d(all_vals[k]) = weights[k]
            # For boundary: all_vals[k] = 1 - r, d(1-r)/d(pts[i]) = -pts[i]/r
            # So contribution to grad = -weights[k] * (-pts[i]/r) = weights[k] * pts[i]/r
            # That's what I have. Good.
            pass
        idx += 1
    
    return f_val, grad


def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.125
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
    
    if len(points) < n:
        # Try smaller spacing
        s = 2 * 0.11
        points = []
        max_range = int(2.0 / s) + 3
        for j in range(-max_range, max_range + 1):
            for i in range(-max_range, max_range + 1):
                x = i * s + (j % 2) * s / 2.0
                y = j * s * np.sqrt(3) / 2.0
                if np.hypot(x, y) <= 0.95:
                    points.append([x, y])
    
    if len(points) == 0:
        return _get_random_initialization(seed=42)
    
    points = np.array(points)
    if len(points) > n:
        # Sort by radius, take closest to center
        radii = np.sqrt(np.sum(points ** 2, axis=1))
        idx = np.argsort(radii, kind='stable')[:n]
        points = points[idx]
    
    return points


def _get_ring_initialization(n):
    """Initialize with concentric rings."""
    # Place points on rings
    points = []
    # Center point
    points.append([0.0, 0.0])
    remaining = n - 1
    
    # Try different ring configurations
    ring_radii = [0.3, 0.55, 0.78, 0.95]
    ring_counts = [1, 6, 12, 18]
    
    # Adjust to fit n points
    used = 0
    for r, cnt in zip(ring_radii, ring_counts):
        take = min(cnt, remaining)
        if take == 0:
            continue
        for k in range(take):
            angle = 2 * np.pi * k / take
            points.append([r * np.cos(angle), r * np.sin(angle)])
        remaining -= take
        used += take
    
    # If we have too many or too few, adjust
    while len(points) < n:
        # Add more on outer ring
        r = 0.9
        k = len(points)
        angle = 2 * np.pi * k / 30
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    points = np.array(points[:n])
    return points


def _get_random_initialization(seed=None):
    """Generate random initial configuration."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x ** 2 + y ** 2 <= 0.9 ** 2:
                pts[i] = [x, y]
                break
    return pts


def _optimize_from_start(x0, beta_schedule, deadline):
    """Run L-BFGS-B optimization with increasing beta."""
    x = x0.copy()
    best_f = _compute_score(x.reshape(N, 2))
    best_x = x.copy()
    
    for beta in beta_schedule:
        if time.time() > deadline:
            break
        
        result = minimize(
            _objective_and_gradient,
            x,
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 2000,
                'maxfun': 5000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'disp': False
            }
        )
        
        if result.success or result.fun < 0:
            x = result.x
            score = _compute_score(x.reshape(N, 2))
            if score > best_f:
                best_f = score
                best_x = x.copy()
    
    return best_x, best_f


def _polish(points, deadline):
    """Final high-precision polish using very high beta."""
    x = points.flatten()
    best_x = x.copy()
    best_score = _compute_score(points)
    
    # Use very high beta for near-exact min
    for beta in [1000, 3000, 5000]:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective_and_gradient,
                x,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 3000,
                    'maxfun': 10000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'disp': False
                }
            )
            if result.success or result.fun < 0:
                x = result.x
                score = _compute_score(x.reshape(N, 2))
                if score > best_score:
                    best_score = score
                    best_x = x.copy()
        except Exception:
            continue
    
    return best_x.reshape(N, 2), best_score


def _local_search(points, deadline, n_restarts=5):
    """Local search with random perturbations."""
    best_pts = points.copy()
    best_score = _compute_score(points)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Perturb
        rng = np.random.default_rng(restart + 100)
        x = best_pts.flatten() + rng.normal(0, 0.02, size=2 * N)
        
        # Clamp to disk
        pts = x.reshape(N, 2)
        for i in range(N):
            r = np.sqrt(np.sum(pts[i] ** 2))
            if r > 0.98:
                pts[i] *= 0.98 / r
        x = pts.flatten()
        
        # Optimize
        beta_schedule = [100, 500, 2000]
        x_opt, score = _optimize_from_start(x, beta_schedule, deadline)
        if score > best_score:
            best_score = score
            best_pts = x_opt.reshape(N, 2)
    
    return best_pts, best_score


def construct():
    """Construct centres of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Multiple initializations
    inits = []
    inits.append(_get_initial_hex_lattice(N))
    inits.append(_get_ring_initialization(N))
    for seed in [1, 2, 3]:
        inits.append(_get_random_initialization(seed=seed))
    
    best_pts = None
    best_score = 0.0
    
    for init_pts in inits:
        if time.time() > deadline:
            break
        
        # Validate initial
        score = _compute_score(init_pts)
        if score <= 0:
            continue
        
        if best_pts is None or score > best_score:
            best_score = score
            best_pts = init_pts.copy()
    
    if best_pts is None:
        best_pts = _get_random_initialization(seed=0)
        best_score = _compute_score(best_pts)
    
    # Main optimization with beta annealing
    beta_schedule = [50, 100, 200, 500, 1000]
    best_pts, best_score = _optimize_from_start(best_pts.flatten(), beta_schedule, deadline)
    best_pts = best_pts.reshape(N, 2)
    
    # Local search
    if time.time() < deadline - 20:
        best_pts, best_score = _local_search(best_pts, deadline, n_restarts=3)
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        best_pts, best_score = _polish(best_pts, deadline)
    
    # Final validation and clamping
    for i in range(N):
        r = np.sqrt(np.sum(best_pts[i] ** 2))
        if r > 0.999:
            best_pts[i] *= 0.999 / r
    
    final_score = _compute_score(best_pts)
    if final_score <= 0:
        best_pts = _get_initial_hex_lattice(N)
    
    return best_pts
