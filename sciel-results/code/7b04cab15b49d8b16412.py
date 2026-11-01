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
    dists_to_origin = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_margin = 1.0 - np.max(dists_to_origin)
    if boundary_margin <= 0:
        return 0.0
    # Pairwise distances
    d = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    if min_pair <= 0:
        return 0.0
    return min(min_pair / 2.0, boundary_margin)


def _soft_min(values, beta):
    """Compute a smooth approximation of the minimum of values."""
    return -np.log(np.sum(np.exp(-beta * values))) / beta


def _objective_and_gradient(x, beta=100.0):
    """
    Objective: negative of soft-min of (pairwise distances / 2, boundary margins).
    Returns (value, gradient).
    """
    pts = x.reshape(N, 2)
    
    # Boundary margins
    dists = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_margins = 1.0 - dists
    
    # Pairwise distances
    d = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    pair_vals = d / 2.0
    
    # Combine all values
    all_vals = np.concatenate([boundary_margins, pair_vals[np.triu_indices(N, k=1)]])
    
    # Soft min
    exp_vals = np.exp(-beta * all_vals)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        sum_exp = 1e-300
    soft_min_val = -np.log(sum_exp) / beta
    weights = exp_vals / sum_exp
    
    # Gradient with respect to points
    grad_pts = np.zeros((N, 2))
    
    # Boundary gradient contribution
    for i in range(N):
        r = dists[i]
        if r > 1e-10:
            unit_vec = pts[i] / r
        else:
            unit_vec = np.array([0.0, 0.0])
        # d(boundary_margin_i)/d(pts_i) = -unit_vec
        grad_pts[i] += -weights[i] * unit_vec
    
    # Pairwise distance gradient contribution
    for i in range(N):
        for j in range(i + 1, N):
            w = weights[N + i * (N - 1) // 2 - i * (i - 1) // 2 + (j - i - 1)]
            # Actually let's index properly
            # The pair (i,j) with i<j is at index: N + (number of pairs before this)
            # Pairs before (i,j): all (a,b) with a<b and (a,b) < (i,j) lexicographically
            # That's sum_{a=0}^{i-1} (N-1-a) + (j-1-i) = i*(N-1) - i*(i-1)//2 + j-1-i
            # Let me just compute it directly
            idx_in_pairs = 0
            for a in range(N):
                for b in range(a + 1, N):
                    if a == i and b == j:
                        break
                    idx_in_pairs += 1
                else:
                    continue
                break
            w = weights[N + idx_in_pairs]
            
            diff = pts[i] - pts[j]
            dist = np.sqrt(np.sum(diff ** 2))
            if dist > 1e-10:
                unit_diff = diff / dist
            else:
                unit_diff = np.array([0.0, 0.0])
            # d(dist_ij)/d(pts_i) = unit_diff, d(dist_ij)/d(pts_j) = -unit_diff
            # d(pair_val)/d(pts_i) = unit_diff / 2
            grad_pts[i] += w * unit_diff / 2.0
            grad_pts[j] += -w * unit_diff / 2.0
    
    # Flatten gradient
    grad_x = grad_pts.flatten()
    
    # Penalty for points outside disk
    outside = np.maximum(0.0, dists - 1.0)
    penalty = 10.0 * np.sum(outside ** 2)
    if np.any(outside > 0):
        for i in range(N):
            if outside[i] > 0:
                if dists[i] > 1e-10:
                    unit_vec = pts[i] / dists[i]
                else:
                    unit_vec = np.array([0.0, 0.0])
                grad_pts[i] += 20.0 * outside[i] * unit_vec
        grad_x = grad_pts.flatten()
        return soft_min_val + penalty, grad_x
    
    return soft_min_val, grad_x


def _get_hex_lattice():
    """Generate a hexagonal lattice initial configuration."""
    best_pts = None
    for trial in range(30):
        spacing = 2.0 / (np.sqrt(N) + 1.0 + 0.02 * trial)
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x ** 2 + y ** 2 <= (1.0 - spacing / 2.0) ** 2:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= N:
            radii = np.sqrt(np.sum(pts ** 2, axis=1))
            idx = np.argsort(radii, kind="stable")[:N]
            candidate = pts[idx]
            if best_pts is None or _compute_score(candidate) > _compute_score(best_pts):
                best_pts = candidate
    if best_pts is not None:
        return best_pts
    return _get_random_initialization(0)


def _get_ring_initialization():
    """Generate a ring-based initial configuration."""
    pts = np.zeros((N, 2))
    # Center point
    pts[0] = [0, 0]
    # Remaining points in rings
    remaining = N - 1
    ring_idx = 1
    angle_offset = 0
    while remaining > 0:
        # Number of points in this ring
        n_in_ring = min(remaining, int(2 * np.pi * ring_idx / 0.3))
        for k in range(n_in_ring):
            idx = 1 + (ring_idx - 1) * 6 + k
            if idx >= N:
                break
            angle = angle_offset + 2 * np.pi * k / n_in_ring
            r = ring_idx * 0.15
            pts[idx] = [r * np.cos(angle), r * np.sin(angle)]
        remaining -= n_in_ring
        ring_idx += 1
        angle_offset += 0.3
    return pts


def _get_random_initialization(seed):
    """Generate random initial configuration."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x ** 2 + y ** 2 <= 0.85 ** 2:
                pts[i] = [x, y]
                break
    return pts


def _optimize_from_start(x0, beta=100.0, maxiter=2000, maxfev=5000):
    """Run L-BFGS-B optimization from a starting point."""
    try:
        result = minimize(
            _objective_and_gradient,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'maxfev': maxfev,
                'gtol': 1e-6,
                'ftol': 1e-8
            }
        )
        return result.x
    except Exception:
        return x0


def _polish(points, beta=500.0, maxiter=1000, maxfev=3000):
    """Polish a configuration with higher beta."""
    x0 = points.flatten()
    x_opt = _optimize_from_start(x0, beta=beta, maxiter=maxiter, maxfev=maxfev)
    pts = x_opt.reshape(N, 2)
    # Ensure inside disk
    r = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(N):
        if r[i] > 0.999:
            scale = 0.999 / r[i]
            pts[i] *= scale
    return pts


def _local_search(points, step_size=0.01, max_iters=100):
    """Simple local search to refine the solution."""
    best_pts = points.copy()
    best_score = _compute_score(best_pts)
    rng = np.random.default_rng(42)
    
    for _ in range(max_iters):
        # Try small random perturbations
        for _ in range(10):
            trial = best_pts.copy()
            trial += rng.normal(0, step_size, size=trial.shape)
            # Clamp to disk
            r = np.sqrt(np.sum(trial ** 2, axis=1))
            for i in range(N):
                if r[i] > 0.999:
                    scale = 0.999 / r[i]
                    trial[i] *= scale
            
            score = _compute_score(trial)
            if score > best_score:
                best_score = score
                best_pts = trial.copy()
    
    return best_pts


def _final_high_precision_polish(points, beta=2000.0, maxiter=500, maxfev=1000):
    """Final high-precision polish with very high beta."""
    x0 = points.flatten()
    
    def obj_grad(x):
        return _objective_and_gradient(x, beta=beta)
    
    try:
        result = minimize(
            obj_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'maxfev': maxfev,
                'gtol': 1e-9,
                'ftol': 1e-12
            }
        )
        pts = result.x.reshape(N, 2)
    except Exception:
        pts = points.copy()
    
    # Ensure inside disk
    r = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(N):
        if r[i] > 0.9999:
            scale = 0.9999 / r[i]
            pts[i] *= scale
    
    return pts


def construct():
    """Construct centres of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_get_hex_lattice())
    inits.append(_get_ring_initialization())
    for seed in range(5):
        inits.append(_get_random_initialization(seed))
    
    best_pts = None
    best_score = 0.0
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        # Optimize from this initialization with increasing beta
        current = init_pts.flatten()
        
        # Stage 1: low beta
        current = _optimize_from_start(current, beta=50.0, maxiter=1000, maxfev=3000)
        pts = current.reshape(N, 2)
        r = np.sqrt(np.sum(pts ** 2, axis=1))
        for i in range(N):
            if r[i] > 0.99:
                scale = 0.99 / r[i]
                pts[i] *= scale
        
        # Stage 2: medium beta
        if time.time() < deadline - 15:
            pts = _polish(pts, beta=200.0, maxiter=1000, maxfev=3000)
        
        # Stage 3: high beta
        if time.time() < deadline - 10:
            pts = _polish(pts, beta=500.0, maxiter=800, maxfev=2000)
        
        # Local search
        if time.time() < deadline - 8:
            pts = _local_search(pts, step_size=0.005, max_iters=50)
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Final high-precision polish
    if best_pts is not None and time.time() < deadline - 5:
        best_pts = _final_high_precision_polish(best_pts, beta=2000.0, maxiter=300, maxfev=500)
        final_score = _compute_score(best_pts)
        if final_score > best_score:
            best_score = final_score
    
    if best_pts is None:
        best_pts = _get_hex_lattice()
    
    return best_pts
