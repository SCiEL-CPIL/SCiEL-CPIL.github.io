"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_radius(points):
    """Feasible common radius: min of boundary clearance and half min pairwise distance."""
    if points is None or len(points) < 2:
        return 0.0
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dist_boundary))
    d = pdist(points)
    min_pair = float(np.min(d)) / 2.0
    return min(min_boundary, min_pair)


def _hexagonal_lattice(spacing):
    """Hexagonal lattice points inside the unit disk (with margin), closest to centre."""
    k = int(1.0 / spacing) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _concentric_rings(n_rings):
    """Concentric rings packing; returns points or None."""
    if n_rings <= 0:
        return None
    if n_rings == 1:
        ring_counts = [N]
    else:
        remaining = N - 1
        base = remaining // (n_rings - 1)
        extra = remaining % (n_rings - 1)
        ring_counts = [1] + [base + (1 if i < extra else 0) for i in range(n_rings - 1)]
    pts = []
    if ring_counts[0] > 0:
        pts.append([0.0, 0.0])
    for ring_idx in range(1, n_rings):
        count = ring_counts[ring_idx]
        if count <= 0:
            continue
        ring_radius = ring_idx / n_rings * 0.95
        for i in range(count):
            angle = 2 * np.pi * i / count
            pts.append([ring_radius * np.cos(angle), ring_radius * np.sin(angle)])
    if len(pts) != N:
        return None
    return np.array(pts)


def _square_grid(spacing):
    """Square grid points inside the unit disk, closest to centre."""
    k = int(1.0 / spacing) + 3
    pts = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = i * spacing
            y = j * spacing
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _refine_local(points, deadline, max_iters=2000, step_size=0.02):
    """Hill-climbing local refinement; respects the deadline."""
    if points is None:
        return None, 0.0
    points = points.copy()
    best_points = points.copy()
    best_r = _compute_radius(points)
    rng = np.random.default_rng(12345)
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        idx = rng.integers(0, len(points))
        original = points[idx].copy()
        best_local = original
        best_local_r = best_r
        for _ in range(6):
            if time.time() > deadline:
                break
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            points[idx] = original + np.array([dx, dy])
            r = _compute_radius(points)
            if r > best_local_r:
                best_local_r = r
                best_local = points[idx].copy()
        points[idx] = best_local
        if best_local_r > best_r:
            best_r = best_local_r
            best_points = points.copy()
    return best_points, best_r


def _soft_min_obj(x, alpha=50.0):
    """Smooth objective: -soft_min(pairwise dist) + boundary penalty."""
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    d_flat = d[d < np.inf]
    if len(d_flat) == 0:
        return 0.0
    
    d_min = np.min(d_flat)
    # Soft min via log-sum-exp
    log_sum = np.log(np.sum(np.exp(-alpha * (d_flat - d_min))))
    soft_min = d_min - log_sum / alpha
    
    # Boundary penalty
    norms = np.hypot(pts[:, 0], pts[:, 1])
    boundary_violation = np.maximum(0.0, norms - (1.0 - soft_min / 2.0))
    boundary_pen = np.sum(boundary_violation ** 2)
    
    return -soft_min + 10.0 * boundary_pen


def _soft_min_grad(x, alpha=50.0):
    """Gradient of soft_min_obj."""
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    d_flat = d[d < np.inf]
    if len(d_flat) == 0:
        return np.zeros_like(x)
    
    d_min = np.min(d_flat)
    
    # Soft min: f = d_min - log(sum(exp(-alpha*(d - d_min)))) / alpha
    # grad f w.r.t. p_i = sum_j w_ij * (p_i - p_j) / d_ij / w_sum
    exp_terms = np.exp(-alpha * (d - d_min))
    np.fill_diagonal(exp_terms, 0.0)
    w_sum = np.sum(exp_terms)
    
    if w_sum == 0:
        return np.zeros_like(x)
    
    grad_pts = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w_ij = exp_terms[i, j]
            d_ij = d[i, j]
            if d_ij > 1e-10:
                diff = pts[i] - pts[j]
                grad_pts[i] += w_ij * diff / d_ij
                grad_pts[j] -= w_ij * diff / d_ij
    
    grad_pts /= w_sum
    grad_soft_min = -grad_pts  # because obj = -soft_min
    
    # Boundary penalty gradient
    norms = np.hypot(pts[:, 0], pts[:, 1])
    # Recompute soft_min for boundary threshold
    d_flat2 = d[d < np.inf]
    d_min2 = np.min(d_flat2)
    log_sum2 = np.log(np.sum(np.exp(-alpha * (d_flat2 - d_min2))))
    soft_min2 = d_min2 - log_sum2 / alpha
    
    boundary_violation = np.maximum(0.0, norms - (1.0 - soft_min2 / 2.0))
    
    grad_boundary = np.zeros_like(pts)
    for i in range(n):
        if boundary_violation[i] > 0:
            norm_i = norms[i]
            if norm_i > 1e-10:
                grad_boundary[i] = 2.0 * boundary_violation[i] * pts[i] / norm_i
    
    grad_obj = grad_soft_min + 10.0 * grad_boundary
    
    return grad_obj.flatten()


def _lbfgs_refine(points, deadline, alpha=50.0, max_iter=300):
    """Use L-BFGS-B with soft-min surrogate to refine the packing."""
    if points is None:
        return None, 0.0
    
    x_init = points.flatten().copy()
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    remaining = deadline - time.time()
    if remaining < 5.0:
        return points.copy(), _compute_radius(points)
    
    try:
        result = minimize(
            _soft_min_obj,
            x_init,
            jac=lambda x: _soft_min_grad(x, alpha=alpha),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                     'maxfun': max_iter * 2, 'maxls': 50}
        )
        
        candidate_pts = result.x.reshape(-1, 2)
        candidate_r = _compute_radius(candidate_pts)
        
        if candidate_r > 0:
            return candidate_pts, candidate_r
        else:
            return points.copy(), _compute_radius(points)
    except Exception:
        return points.copy(), _compute_radius(points)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s wall-clock limit

    candidates = []

    # Hexagonal lattice: try a range of spacings and keep the best feasible.
    for trial in range(40):
        spacing = 0.05 + trial * 0.005
        pts = _hexagonal_lattice(spacing)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    # Concentric rings: a few ring counts.
    for n_rings in range(3, 20):
        pts = _concentric_rings(n_rings)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    # Square grid: a range of spacings.
    for trial in range(30):
        spacing = 0.05 + trial * 0.005
        pts = _square_grid(spacing)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    if not candidates:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_points, best_r = candidates[0]

    # Phase 1: L-BFGS-B refinement with soft-min surrogate (coordinated moves)
    # Try the top candidates with increasing alpha for finer tuning
    for cand_idx in range(min(3, len(candidates))):
        if time.time() > deadline - 30.0:
            break
        cand_pts, cand_r = candidates[cand_idx]
        
        # First pass with alpha=50
        refined_pts, refined_r = _lbfgs_refine(cand_pts, deadline, alpha=50.0, max_iter=200)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r
        
        # Second pass with higher alpha for finer tuning, starting from improved solution
        if time.time() > deadline - 15.0:
            break
        refined_pts2, refined_r2 = _lbfgs_refine(best_points, deadline, alpha=100.0, max_iter=150)
        if refined_pts2 is not None and refined_r2 > best_r:
            best_points, best_r = refined_pts2, refined_r2

    # Phase 2: Local hill-climbing polish (single-point moves for final precision)
    if time.time() < deadline - 5.0:
        refined_pts, refined_r = _refine_local(best_points, deadline,
                                               max_iters=3000, step_size=0.01)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    return best_points

# EVOLVE-BLOCK-END
